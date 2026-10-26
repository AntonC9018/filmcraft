//! Speech-to-text through a Handy local transcription server (loopback HTTP).
//!
//! [Handy](https://github.com/cjpais/Handy) is a desktop dictation app whose optional local
//! server takes up to five minutes of 16 kHz mono Float32-LE PCM per `POST /transcribe` and
//! answers with the recognised text, plus `segments` with time bounds when the underlying engine
//! provides them. [`HandyTranscriber`] speaks that protocol: it splits longer audio at quiet
//! windows so no word is cut, spreads the words over each chunk's (or segment's) span, tightens
//! the bounds with [`crate::vad`] and labels speakers with [`crate::diarize`], like the Whisper
//! engine does — so transcripts from a Handy-hosted model behave like local ones.
//!
//! The server is authenticated with a bearer token and binds IPv4 loopback only, rejecting
//! requests whose `Host` is anything but `127.0.0.1:<port>`; the URL must therefore name
//! `127.0.0.1`. Configuration comes from the environment ([`Config::from_env`]):
//! `FILMCRAFT_HANDY_URL` (default `http://127.0.0.1:8003`), `FILMCRAFT_HANDY_TOKEN` or
//! `FILMCRAFT_HANDY_TOKEN_FILE` (a 0600 file with the token), and optional
//! `FILMCRAFT_HANDY_MODEL` (recorded in the transcript's `source`).

use std::time::{Duration, Instant};

use filmcraft_project::{Transcript, Word};
use serde::Deserialize;

use crate::{Options, ProgressFn, SAMPLE_RATE, SpeechError, Transcriber, sample_tick};

/// The server accepts at most five minutes per request; keep headroom for the split search.
const MAX_CHUNK: usize = SAMPLE_RATE as usize * 290;
/// How long to wait when another client's transcription holds the server's single slot.
const BUSY_TIMEOUT: Duration = Duration::from_secs(120);
const BUSY_RETRY: Duration = Duration::from_secs(2);

/// Connection settings for a Handy local server.
#[derive(Clone, Debug, PartialEq)]
pub struct Config {
    /// `http://127.0.0.1:<port>` — the server rejects any other `Host`.
    pub url: String,
    /// The bearer token (`Authorization: Bearer <token>`).
    pub token: String,
    /// Model id recorded in the transcript's `source` (`handy:<model>`); the server keeps its
    /// own selected model, the client never switches it.
    pub model: Option<String>,
}

impl Config {
    /// Read the `FILMCRAFT_HANDY_*` environment. `None` = unconfigured (the feature stays off);
    /// `Err` = configured wrongly (surfaced by the host as a warning).
    pub fn from_env() -> Result<Option<Config>, String> {
        let token = match std::env::var("FILMCRAFT_HANDY_TOKEN") {
            Ok(t) if !t.trim().is_empty() => t.trim().to_string(),
            _ => match std::env::var("FILMCRAFT_HANDY_TOKEN_FILE") {
                Ok(f) if !f.trim().is_empty() => std::fs::read_to_string(f.trim()).map_err(|e| format!("FILMCRAFT_HANDY_TOKEN_FILE: {e}"))?.trim().to_string(),
                _ => return Ok(None),
            },
        };
        if token.is_empty() {
            return Err("FILMCRAFT_HANDY_TOKEN is empty".into());
        }
        let url = std::env::var("FILMCRAFT_HANDY_URL").unwrap_or_else(|_| "http://127.0.0.1:8003".into());
        let url = loopback_url(&url)?;
        let model = std::env::var("FILMCRAFT_HANDY_MODEL").ok().filter(|m| !m.trim().is_empty());
        Ok(Some(Config { url, token, model: model.map(|m| m.trim().to_string()) }))
    }
}

/// Accept only the server's literal IPv4 authority, with no user-info, path,
/// query or fragment. A prefix check also accepts `127.0.0.1:8003@remote-host`.
fn loopback_url(url: &str) -> Result<String, String> {
    let url = url.trim().trim_end_matches('/');
    let valid = url
        .strip_prefix("http://127.0.0.1:")
        .is_some_and(|port| !port.is_empty() && port.bytes().all(|b| b.is_ascii_digit()) && port.parse::<u16>().is_ok_and(|p| p > 0));
    if !valid {
        return Err("FILMCRAFT_HANDY_URL must be http://127.0.0.1:<port> with no other URL components".into());
    }
    Ok(url.to_string())
}

/// A [`Transcriber`] that sends audio to a Handy local server and maps the reply to a
/// [`Transcript`]. One request runs at a time on the server; concurrent requests wait.
pub struct HandyTranscriber {
    url: String,
    token: String,
    id: String,
}

impl HandyTranscriber {
    pub fn new(config: &Config) -> Self {
        let id = match &config.model {
            Some(m) => format!("handy:{m}"),
            None => "handy".into(),
        };
        Self { url: config.url.clone(), token: config.token.clone(), id }
    }
}

#[derive(Debug, Deserialize)]
struct Reply {
    #[serde(default)]
    text: String,
    #[serde(default)]
    segments: Vec<Segment>,
}

#[derive(Debug, Deserialize)]
struct Segment {
    start: f64,
    end: f64,
    #[serde(default)]
    text: String,
}

impl Transcriber for HandyTranscriber {
    fn id(&self) -> String {
        self.id.clone()
    }

    fn transcribe(&self, audio: &[f32], opts: &Options, progress: ProgressFn) -> Result<Transcript, SpeechError> {
        let agent = agent();
        let chunks = chunk_bounds(audio);
        let mut words = Vec::new();
        for (i, r) in chunks.iter().enumerate() {
            let done = i as f32 / chunks.len() as f32;
            if !progress(done, "Transcribing (Handy)") {
                return Err(SpeechError::Cancelled);
            }
            let reply = self.post(&agent, &audio[r.clone()], opts, progress, done)?;
            if reply.segments.is_empty() {
                spread_words(&reply.text, r.start, r.end, &mut words);
            } else {
                let before = words.len();
                for s in &reply.segments {
                    // Treat server timings as untrusted. Clamp each endpoint separately
                    // before adding the chunk offset; a start beyond the chunk must not
                    // become the lower bound of a clamp whose upper bound is smaller.
                    if !s.start.is_finite() || !s.end.is_finite() {
                        continue;
                    }
                    let len = r.len() as f64;
                    let a = (s.start * SAMPLE_RATE as f64).clamp(0.0, len) as usize;
                    let b = (s.end * SAMPLE_RATE as f64).clamp(0.0, len) as usize;
                    spread_words(&s.text, r.start + a, r.start + b, &mut words);
                }
                // every segment was degenerate: fall back to the whole chunk
                if words.len() == before && !reply.text.is_empty() {
                    spread_words(&reply.text, r.start, r.end, &mut words);
                }
            }
        }
        let mut t = Transcript { language: opts.language.clone().unwrap_or_default(), source: self.id.clone(), speakers: Vec::new(), words };
        t.normalize();
        crate::vad::tighten_words(audio, &mut t.words);
        if opts.diarize && !t.words.is_empty() {
            if !progress(0.96, "Labelling speakers") {
                return Err(SpeechError::Cancelled);
            }
            let p = crate::diarize::Params { max_speakers: opts.max_speakers, ..Default::default() };
            crate::diarize::diarize(audio, &mut t, &p);
        }
        progress(1.0, "Done");
        Ok(t)
    }
}

impl HandyTranscriber {
    /// One chunk to `POST /transcribe`, waiting out `speech_busy` replies.
    fn post(&self, agent: &ureq::Agent, chunk: &[f32], opts: &Options, progress: ProgressFn, done: f32) -> Result<Reply, SpeechError> {
        // Config is public, so validate here too, before any credentials or audio
        // can leave the process (not only when loading environment settings).
        let url = loopback_url(&self.url).map_err(SpeechError::Model)?;
        // The server rejects any sample outside ±1 (and non-finite ones) for the whole
        // request, so decoded media that clips or glitches must not pass through raw.
        let mut body = Vec::with_capacity(chunk.len() * 4);
        for s in chunk {
            let s = if s.is_finite() { s.clamp(-1.0, 1.0) } else { 0.0 };
            body.extend_from_slice(&s.to_le_bytes());
        }
        let language = server_language(opts.language.as_deref());
        let deadline = Instant::now() + BUSY_TIMEOUT;
        loop {
            let attempt = agent
                .post(format!("{url}/transcribe"))
                .header("Authorization", format!("Bearer {}", self.token))
                .header("Content-Type", "application/octet-stream")
                .header("X-Speech-Language", language)
                .send(body.as_slice());
            let response = attempt.map_err(|e| unavailable(&self.url, &e))?;
            match response.status().as_u16() {
                200 => {
                    let text = response.into_body().read_to_string().map_err(|e| SpeechError::Model(e.to_string()))?;
                    return serde_json::from_str::<Reply>(&text).map_err(|e| SpeechError::Model(format!("Handy reply is not valid JSON: {e}")));
                }
                409 if Instant::now() < deadline => {
                    if !progress(done, "Waiting for the Handy server (busy)") {
                        return Err(SpeechError::Cancelled);
                    }
                    std::thread::sleep(BUSY_RETRY);
                }
                401 => {
                    return Err(SpeechError::Model("Handy rejected the bearer token (check FILMCRAFT_HANDY_TOKEN)".into()));
                }
                status => {
                    let detail = response.into_body().read_to_string().map(|s| s.chars().take(200).collect::<String>()).unwrap_or_default();
                    return Err(SpeechError::Model(format!("Handy answered {status}: {detail}")));
                }
            }
        }
    }
}

fn unavailable(url: &str, e: &ureq::Error) -> SpeechError {
    SpeechError::Unavailable(format!("the Handy speech server at {url} did not answer ({e}); start it and retry"))
}

/// The server accepts 2–16 ASCII letters and hyphens; anything else would be rejected, so fall
/// back to auto-detection instead of failing the request.
fn server_language(language: Option<&str>) -> &str {
    match language {
        Some(l) if (2..=16).contains(&l.len()) && l.bytes().all(|b| b.is_ascii_alphabetic() || b == b'-') => l,
        _ => "auto",
    }
}

fn agent() -> ureq::Agent {
    ureq::Agent::config_builder()
        // Handle protocol statuses ourselves, including the server's busy reply.
        .http_status_as_error(false)
        // Audio and credentials belong only on the configured loopback socket.
        .proxy(None)
        .max_redirects(0)
        // Connecting to a down server should fail fast; transcription itself can run minutes.
        .timeout_connect(Some(Duration::from_secs(5)))
        .build()
        .into()
}

/// Sample ranges to send, each within the server's limit. Splits land on the quietest 0.3 s
/// window in the half minute before the limit, so a running word is not cut in two.
fn chunk_bounds(audio: &[f32]) -> Vec<std::ops::Range<usize>> {
    if audio.len() <= MAX_CHUNK {
        return std::iter::once(0..audio.len()).collect();
    }
    let frame = (SAMPLE_RATE / 100) as usize; // matches vad's 10 ms frames
    let db = crate::vad::frame_db(audio);
    let quiet = 30; // 0.3 s of near-silence
    let search = 30 * 100; // split somewhere in the last half minute of the chunk
    let mut bounds = Vec::new();
    let mut start = 0;
    while audio.len() - start > MAX_CHUNK {
        let limit = start + MAX_CHUNK;
        let hi = (limit / frame).min(db.len());
        let lo = hi.saturating_sub(search).max(start / frame + 1);
        let (mut best, mut best_level) = (limit, f32::INFINITY);
        for a in lo..hi.saturating_sub(quiet).max(lo) {
            let level = db[a..a + quiet].iter().sum::<f32>();
            if level < best_level {
                best_level = level;
                best = (a + quiet / 2) * frame;
            }
        }
        let split = best.clamp(start + MAX_CHUNK / 2, limit);
        bounds.push(start..split);
        start = split;
    }
    bounds.push(start..audio.len());
    bounds
}

/// Lay the words of `text` over the sample range `[from, to)`, each word's share proportional to
/// its length — server segments are sentence-level at best, this keeps word bounds plausible
/// until [`crate::vad::tighten_words`] snaps them onto the actual speech.
fn spread_words(text: &str, from: usize, to: usize, out: &mut Vec<Word>) {
    let words: Vec<&str> = text.split_whitespace().collect();
    if words.is_empty() || to <= from {
        return;
    }
    let total: usize = words.iter().map(|w| w.chars().count()).sum();
    let span = (to - from) as f64;
    let mut at = from;
    for w in words {
        let share = (span * w.chars().count() as f64 / total.max(1) as f64).round() as usize;
        let start = at;
        let end = (start + share).min(to);
        at = end;
        out.push(Word::new(w, sample_tick(start as i64), sample_tick(end as i64)));
    }
    // rounding can leave the last word just short of the range end
    if let Some(last) = out.last_mut() {
        last.end = sample_tick(to as i64);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::seconds_tick;

    /// A real HTTP seam using synthetic credentials and audio, never the live server.
    fn server(replies: Vec<(u16, String)>) -> (HandyTranscriber, std::thread::JoinHandle<Vec<Vec<u8>>>) {
        use std::io::{BufRead, BufReader, Read, Write};
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let url = format!("http://{}", listener.local_addr().unwrap());
        let worker = std::thread::spawn(move || {
            let mut bodies = Vec::new();
            for (status, body) in replies {
                let (mut stream, _) = listener.accept().unwrap();
                stream.set_read_timeout(Some(Duration::from_secs(5))).unwrap();
                let mut reader = BufReader::new(&mut stream);
                let mut line = String::new();
                reader.read_line(&mut line).unwrap();
                assert_eq!(line.trim(), "POST /transcribe HTTP/1.1");
                let mut length = 0;
                let mut authorized = false;
                loop {
                    line.clear();
                    reader.read_line(&mut line).unwrap();
                    if line == "\r\n" {
                        break;
                    }
                    if let Some((key, value)) = line.split_once(':') {
                        if key.eq_ignore_ascii_case("content-length") {
                            length = value.trim().parse().unwrap();
                        }
                        if key.eq_ignore_ascii_case("authorization") {
                            authorized = value.trim() == "Bearer synthetic-token";
                        }
                    }
                }
                assert!(authorized);
                let mut audio = vec![0; length];
                reader.read_exact(&mut audio).unwrap();
                bodies.push(audio);
                write!(stream, "HTTP/1.1 {status} Reply\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}", body.len()).unwrap();
            }
            bodies
        });
        (HandyTranscriber::new(&Config { url, token: "synthetic-token".into(), model: None }), worker)
    }

    #[test]
    fn hostile_segment_bounds_stay_inside_the_audio_and_fall_back() {
        for segments in [
            r#"[{"start":2.0,"end":3.0,"text":"outside"}]"#,
            r#"[{"start":0.9,"end":0.1,"text":"reversed"}]"#,
            r#"[{"start":0.0,"end":0.0,"text":"untimed"}]"#,
            r#"[{"start":-2.0,"end":20.0,"text":"fallback words"}]"#,
        ] {
            let body = format!(r#"{{"text":"fallback words","segments":{segments}}}"#);
            let (client, worker) = server(vec![(200, body)]);
            let transcript = client.transcribe(&vec![0.1; 16_000], &Options { diarize: false, ..Default::default() }, &mut |_, _| true).unwrap();
            worker.join().unwrap();
            assert_eq!(transcript.words.iter().map(|w| w.text.as_str()).collect::<Vec<_>>(), ["fallback", "words"]);
            assert!(transcript.words.iter().all(|w| w.start >= sample_tick(0) && w.end <= sample_tick(16_000) && w.end >= w.start));
        }
    }

    #[test]
    fn busy_replies_retry_and_resend_sanitized_audio() {
        let (client, worker) = server(vec![(409, r#"{"error":"speech_busy"}"#.into()), (200, r#"{"text":"ready"}"#.into())]);
        let mut waiting = 0;
        let reply = client
            .post(
                &agent(),
                &[2.0, -2.0, f32::NAN],
                &Options::default(),
                &mut |_, message| {
                    if message.contains("busy") {
                        waiting += 1;
                    }
                    true
                },
                0.0,
            )
            .unwrap();
        assert_eq!(reply.text, "ready");
        assert_eq!(waiting, 1);
        let bodies = worker.join().unwrap();
        let expected: Vec<u8> = [1.0_f32, -1.0, 0.0].into_iter().flat_map(f32::to_le_bytes).collect();
        assert_eq!(bodies, [expected.clone(), expected]);
    }

    #[test]
    fn busy_wait_can_be_cancelled_and_auth_errors_are_actionable() {
        let (client, worker) = server(vec![(409, "{}".into())]);
        assert_eq!(client.post(&agent(), &[0.0], &Options::default(), &mut |_, _| false, 0.0).unwrap_err(), SpeechError::Cancelled);
        worker.join().unwrap();
        let (client, worker) = server(vec![(401, "{}".into())]);
        let err = client.post(&agent(), &[0.0], &Options::default(), &mut |_, _| true, 0.0).unwrap_err();
        assert!(matches!(err, SpeechError::Model(message) if message.contains("bearer token")));
        worker.join().unwrap();
    }

    #[test]
    fn endpoint_is_strictly_loopback_and_never_uses_proxies_or_redirects() {
        assert_eq!(loopback_url(" http://127.0.0.1:8003/ ").unwrap(), "http://127.0.0.1:8003");
        for url in [
            "http://127.0.0.1:8003@remote.invalid",
            "http://127.0.0.1:8003/path",
            "http://127.0.0.1:8003?query",
            "http://127.0.0.1:8003#fragment",
            "http://127.0.0.1:0",
            "http://127.0.0.1:65536",
            "http://127.0.0.1:",
            "https://127.0.0.1:8003",
        ] {
            assert!(loopback_url(url).is_err(), "{url}");
            let client = HandyTranscriber::new(&Config { url: url.into(), token: "synthetic-token".into(), model: None });
            assert!(matches!(client.post(&agent(), &[0.0], &Options::default(), &mut |_, _| true, 0.0), Err(SpeechError::Model(_))));
        }
        assert!(agent().config().proxy().is_none());
        assert_eq!(agent().config().max_redirects(), 0);
    }

    #[test]
    fn language_is_validated_for_the_server() {
        assert_eq!(server_language(Some("ru")), "ru");
        assert_eq!(server_language(Some("zh-CN")), "zh-CN");
        assert_eq!(server_language(Some("auto")), "auto");
        assert_eq!(server_language(None), "auto");
        assert_eq!(server_language(Some("r")), "auto");
        assert_eq!(server_language(Some("russian-language-is-long")), "auto");
    }

    #[test]
    fn words_spread_over_the_whole_chunk_without_segments() {
        let mut words = Vec::new();
        spread_words("раз два три", 0, 16_000, &mut words);
        assert_eq!(words.len(), 3);
        assert_eq!(words[0].start, seconds_tick(0.0));
        assert_eq!(words[2].end, seconds_tick(1.0));
        assert!(words.iter().all(|w| w.end > w.start));
    }

    #[test]
    fn words_spread_within_segments() {
        let mut words = Vec::new();
        spread_words("a b", 16_000, 32_000, &mut words);
        assert!(words[0].start >= seconds_tick(1.0) && words[1].end <= seconds_tick(2.0));
    }

    #[test]
    fn reply_parses_text_and_optional_segments() {
        let plain: Reply = serde_json::from_str(r#"{"text":"привет"}"#).unwrap();
        assert_eq!(plain.text, "привет");
        assert!(plain.segments.is_empty());
        let timed: Reply =
            serde_json::from_str(r#"{"text":"раз два","segments":[{"start":0.0,"end":1.5,"text":"раз"}, {"start":1.5,"end":2.0,"text":"два"}]}"#).unwrap();
        assert_eq!(timed.segments.len(), 2);
        assert_eq!(timed.segments[1].text, "два");
        // engines without times report a single zero-span row
        let untimed: Reply = serde_json::from_str(r#"{"text":"hello","segments":[{"start":0.0,"end":0.0,"text":"hello"}]}"#).unwrap();
        assert_eq!(untimed.segments.len(), 1);
    }

    #[test]
    fn zero_span_words_fall_back_to_the_whole_chunk() {
        let mut words = Vec::new();
        spread_words("", 0, 100, &mut words);
        spread_words("hi", 50, 50, &mut words);
        assert!(words.is_empty());
        spread_words("a b", 0, 16_000, &mut words);
        assert_eq!(words.len(), 2);
        assert_eq!(words[1].end, seconds_tick(1.0));
    }

    #[test]
    fn long_audio_splits_at_quiet_windows_within_the_limit() {
        // 700 s: a loud tone everywhere except quiet windows every ~10 s.
        let n = 700 * SAMPLE_RATE as usize;
        let audio: Vec<f32> = (0..n)
            .map(|i| {
                let t = i as f32 / SAMPLE_RATE as f32;
                let in_quiet = (t % 10.0) > 9.5 && ((t % 10.0) < 9.9);
                if in_quiet { 0.0 } else { 0.2 * (t * 440.0).sin() }
            })
            .collect();
        let bounds = chunk_bounds(&audio);
        assert!(bounds.len() >= 3);
        assert_eq!(bounds[0].start, 0);
        assert_eq!(bounds.last().unwrap().end, n);
        for (i, r) in bounds.iter().enumerate() {
            assert!(r.len() <= MAX_CHUNK, "chunk {i} too long");
            if i > 0 {
                assert_eq!(r.start, bounds[i - 1].end);
                // splits land in the quiet window (0.5 s of silence before the 9.5 s mark)
                let t = r.start as f32 / SAMPLE_RATE as f32;
                assert!((t % 10.0) > 9.5 && (t % 10.0) < 9.95, "split at {t} is not quiet");
            }
        }
    }

    #[test]
    fn short_audio_is_one_chunk() {
        assert_eq!(chunk_bounds(&[0.0; 1000]), vec![0..1000]);
        assert_eq!(chunk_bounds(&[]), vec![0..0]);
    }
}
