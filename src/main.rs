use num::complex::Complex;
use std::vec::Vec;
use std::time::{Duration, Instant};
use std::rc::Rc;
use std::borrow::BorrowMut;
use std::cell::Cell;

use circular_buffer::CircularBuffer;

use pipewire as pw;
use pw::{properties::properties, spa};
use spa::param::format::{MediaSubtype, MediaType};
use spa::param::format_utils;
use spa::pod::Pod;
use std::convert::TryInto;
use std::mem;
use std::thread;

use color_eyre::Result;
use crossterm::event::{self, Event, KeyEvent};
use ratatui::widgets::{Bar, BarChart, BarGroup};
use ratatui::{DefaultTerminal, Frame};

use std::sync::mpsc;
use std::sync::mpsc::{Receiver, Sender};

struct UserData {
    format: spa::param::audio::AudioInfoRaw,
    cursor_move: bool,
}

fn ditfft2(x: &[f32]) -> Vec<Complex<f32>> {
    fn _ditfft2(a: &mut Vec<Complex<f32>>) {
        let n = a.len();
        if n != 1 {
            let mut a0: Vec<Complex<f32>> = vec![Complex::new(0.0, 0.0); n / 2];
            let mut a1: Vec<Complex<f32>> = vec![Complex::new(0.0, 0.0); n / 2];
            for i in 0..n / 2 {
                a0[i] = a[2 * i];
                a1[i] = a[2 * i + 1];
            }

            _ditfft2(&mut a0);
            _ditfft2(&mut a1);

            let ang: f32 = -2.0 * std::f32::consts::PI / (n as f32);
            let mut w: Complex<f32> = Complex::new(1.0, 0.0);
            let wn: Complex<f32> = Complex::new(ang.cos(), ang.sin());
            for k in 0..n / 2 {
                a[k] = a0[k] + w * a1[k];
                a[k + n / 2] = a0[k] - w * a1[k];
                w = w * wn;
            }
        }
    }

    let mut x: Vec<Complex<f32>> = x.iter().map(|x| Complex::new(*x, 0.0)).collect();
    _ditfft2(&mut x);
    x
}

fn decimate(fft: Vec<Complex<f32>>, sample_rate: f32, output_freqs: usize) -> Vec<f32> {
    let fft_re = fft.iter().map(|x| x.norm()).collect::<Vec<f32>>();

    let bin_freq = sample_rate / fft.len() as f32;
    let range = (10000.0 / bin_freq) as usize;

    let mut negative_component: Vec<f32> = fft_re[fft_re.len() - range..fft_re.len()]
        .iter()
        .copied()
        .collect();


    negative_component.extend_from_slice(&fft_re[0..range + 1]);
    let fft = negative_component;

    let mut fft_resampled: Vec<f32> = Vec::new();

    let n = fft.len();
    for i in 0..output_freqs {
        fft_resampled.push(
            100.0
                * (fft[((n / output_freqs) * i)..((n / output_freqs) * (i + 1))]
                    .iter()
                    .sum::<f32>()
                    / ((n / output_freqs) as f32))
                    .abs(),
        );
    }
    
    let L = fft_resampled.len() / 2;
    let mut left = fft_resampled[0..fft_resampled.len() / 2].iter().enumerate().map(|(i,x)| *x*((L - i) as f32)).collect::<Vec<f32>>();
    let mut right = fft_resampled[fft_resampled.len()/2..fft_resampled.len()].iter().enumerate().map(|(i,x)| *x*((i+1) as f32)).collect::<Vec<f32>>();
    
    left.append(&mut right);
    left
}

fn sound_capture(tx: Sender<(Vec<f32>, f32)>) -> Result<(), pw::Error> {
    let mainloop = pw::main_loop::MainLoop::new(None)?;
    let context = pw::context::Context::new(&mainloop)?;
    let core = context.connect(None)?;

    let data = UserData {
        format: Default::default(),
        cursor_move: false,
    };

    let props = properties! {
        *pw::keys::MEDIA_TYPE => "Audio",
        *pw::keys::MEDIA_CATEGORY => "Capture",
        *pw::keys::MEDIA_ROLE => "Music",
    };

    let stream = pw::stream::Stream::new(&core, "audio-capture", props)?;
    let mut sample_buffer = CircularBuffer::<4096, f32>::new();
    let mut start = Instant::now();

    let mut absolute_max: f32 = 1.0;

    let _listener = stream
        .add_local_listener_with_user_data(data)
        .param_changed(|_, user_data, id, param| {
            // NULL means to clear the format
            let Some(param) = param else {
                return;
            };
            if id != pw::spa::param::ParamType::Format.as_raw() {
                return;
            }

            let (media_type, media_subtype) = match format_utils::parse_format(param) {
                Ok(v) => v,
                Err(_) => return,
            };

            // only accept raw audio
            if media_type != MediaType::Audio || media_subtype != MediaSubtype::Raw {
                return;
            }

            // call a helper function to parse the format for us.
            user_data
                .format
                .parse(param)
                .expect("Failed to parse param changed to AudioInfoRaw");
        })
        .process(move |stream, user_data| match stream.dequeue_buffer() {
            None => {}
            Some(mut buffer) => {
                let datas = buffer.datas_mut();
                if datas.is_empty() {
                    return;
                }

                let data = &mut datas[0];
                let n_channels = user_data.format.channels();
                let n_samples = data.chunk().size() / (mem::size_of::<f32>() as u32);

                if let Some(samples) = data.data() {
                    let mut buffer: Vec<Vec<f32>> = Vec::new();
                    for c in 0..n_channels {
                        let mut b: Vec<f32> = Vec::new();
                        for n in (c..n_samples).step_by(n_channels as usize) {
                            let start = n as usize * mem::size_of::<f32>();
                            let end = start + mem::size_of::<f32>();
                            let chan = &samples[start..end];
                            let f = f32::from_le_bytes(chan.try_into().unwrap());
                            b.push(f);
                        }
                        buffer.push(b);
                    }

                    for i in 0..buffer[0].len() {
                        let mut acc = 0.0;
                        for c in 0..buffer.len() {
                            acc += buffer[c][i];
                        }

                        sample_buffer.push_back(acc / buffer.len() as f32);
                    }

                    user_data.cursor_move = true;

                    if start.elapsed() < Duration::from_millis(20)  {
                        return;
                    }
                    else {
                        start = Instant::now();
                    }
                    
                    let fft = ditfft2(&sample_buffer.to_vec());
                    let dec = decimate(fft, 44100.0, 32);
                    let m = dec.iter().max_by(|a, b| a.partial_cmp(b).unwrap()).unwrap();

                    if absolute_max < *m {
                        absolute_max = *m; 
                    }
                    else {
                        absolute_max -= (absolute_max - *m)/1000.0;
                    }
                    let r = tx.send((dec, absolute_max));
                    if r.is_err() {
                        return;
                    }

                }
            }
        })
        .register()?;

    /* Make one parameter with the supported formats. The SPA_PARAM_EnumFormat
     * id means that this is a format enumeration (of 1 value).
     * We leave the channels and rate empty to accept the native graph
     * rate and channels. */
    let mut audio_info = spa::param::audio::AudioInfoRaw::new();
    audio_info.set_format(spa::param::audio::AudioFormat::F32LE);
    let obj = pw::spa::pod::Object {
        type_: pw::spa::utils::SpaTypes::ObjectParamFormat.as_raw(),
        id: pw::spa::param::ParamType::EnumFormat.as_raw(),
        properties: audio_info.into(),
    };
    let values: Vec<u8> = pw::spa::pod::serialize::PodSerializer::serialize(
        std::io::Cursor::new(Vec::new()),
        &pw::spa::pod::Value::Object(obj),
    )
    .unwrap()
    .0
    .into_inner();

    let mut params = [Pod::from_bytes(&values).unwrap()];

    /* Now connect this stream. We ask that our process function is
     * called in a realtime thread. */
    stream.connect(
        spa::utils::Direction::Input,
        None,
        pw::stream::StreamFlags::AUTOCONNECT
            | pw::stream::StreamFlags::MAP_BUFFERS
            | pw::stream::StreamFlags::RT_PROCESS,
        &mut params,
    )?;

    mainloop.run();
    Ok(())
}

struct Ui {
    rx: Receiver<(Vec<f32>, f32)>,
    max: u64,
}

impl Ui {
    fn new(rx: Receiver<(Vec<f32>, f32)>) -> Self {
        Self { rx, max: 1 }
    }

    fn run(&self, mut terminal: DefaultTerminal) -> Result<()> {
        let mut s: u32 = 0;
        loop {
            let (fft, max) = self.rx.recv().unwrap();

            dbg!(max);

            if s < 2 {
                s += 1;
                continue;
            } else {
                s = 0;
            }
            
            terminal.draw(move |frame: &mut Frame| {
                let area = frame.area();

                let mut bars: Vec<Bar> = Vec::new();

                for x in &fft {
                    let value = (1000.0 * *x) as u64;
                    let bar = Bar::default()
                        .value(value)
                        .text_value("".into())
                        .label("".into());
                    bars.push(bar);
                }

                let barchart = BarChart::default()
                    .bar_width(2)
                    .data(BarGroup::default().bars(&bars))
                    .max((1000.0 * max) as u64);
                frame.render_widget(barchart.clone(), area);
            })?;

            if event::poll(std::time::Duration::from_millis(0))? {
                if event::read()?
                    == Event::Key(KeyEvent::new(
                        event::KeyCode::Char('q'),
                        event::KeyModifiers::NONE,
                    ))
                {
                    break ();
                }
            }
        }

        Ok(())
    }
}

fn ui(rx: Receiver<(Vec<f32>, f32)>) -> Result<()> {
    color_eyre::install()?;
    let terminal = ratatui::init();
    let mut ui = Ui::new(rx);
    let result = ui.run(terminal);
    ratatui::restore();
    result

    //let mut max = 1.0;
    //loop {
    //    let fft = rx.recv().unwrap();
    //    for x in &fft {
    //        if *x > max {
    //            max = *x;
    //        }
    //    }
    //    dbg!(&fft);
    //    dbg!(max);
    //}
    //Ok(())
}

fn main() -> Result<(), pw::Error> {
    let (tx, rx): (Sender<(Vec<f32>, f32)>, Receiver<(Vec<f32>, f32)>) = mpsc::channel();

    let sound_capture_thread = thread::spawn(move || {
        sound_capture(tx).unwrap();
    });

    let ui_thread = thread::spawn(move || {
        ui(rx).unwrap();
    });

    sound_capture_thread.join().unwrap();
    ui_thread.join().unwrap();
    Ok(())
}
