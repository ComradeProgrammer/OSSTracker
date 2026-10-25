//! opentrack "UDP over network" protocol: six little-endian f64 values
//! (x, y, z in cm; yaw, pitch, roll in degrees).

use std::net::{SocketAddr, ToSocketAddrs, UdpSocket};

#[derive(Default)]
pub struct UdpSender {
    socket: Option<UdpSocket>,
    target: Option<(String, u16, SocketAddr)>,
}

pub fn encode(pose: &[f64; 6]) -> [u8; 48] {
    // Internal order is yaw, pitch, roll, x, y, z; opentrack wants x, y, z, yaw, pitch, roll.
    let order = [3, 4, 5, 0, 1, 2];
    let mut buf = [0u8; 48];
    for (k, &i) in order.iter().enumerate() {
        buf[k * 8..k * 8 + 8].copy_from_slice(&pose[i].to_le_bytes());
    }
    buf
}

impl UdpSender {
    pub fn send(&mut self, host: &str, port: u16, pose: &[f64; 6]) -> Result<(), String> {
        let addr = match &self.target {
            Some((h, p, a)) if h == host && *p == port => *a,
            _ => {
                let a = (host, port)
                    .to_socket_addrs()
                    .map_err(|e| format!("bad UDP address {host}:{port}: {e}"))?
                    .next()
                    .ok_or_else(|| format!("bad UDP address {host}:{port}"))?;
                self.target = Some((host.to_string(), port, a));
                a
            }
        };
        if self.socket.as_ref().map(|s| s.local_addr().map(|l| l.is_ipv4() != addr.is_ipv4()).unwrap_or(true)).unwrap_or(true) {
            let bind = if addr.is_ipv4() { "0.0.0.0:0" } else { "[::]:0" };
            self.socket = Some(UdpSocket::bind(bind).map_err(|e| format!("cannot open UDP socket: {e}"))?);
        }
        let sock = self.socket.as_ref().expect("socket");
        match sock.send_to(&encode(pose), addr) {
            Ok(_) => Ok(()),
            // Nothing listening yet (opentrack not started) is normal; don't treat it as an error.
            Err(e) if e.kind() == std::io::ErrorKind::ConnectionRefused => Ok(()),
            Err(e) => Err(format!("UDP send failed: {e}")),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn packet_layout_matches_opentrack() {
        let b = encode(&[10.0, 20.0, 30.0, 1.0, 2.0, 3.0]);
        let vals: Vec<f64> = b.chunks(8).map(|c| f64::from_le_bytes(c.try_into().unwrap())).collect();
        assert_eq!(vals, vec![1.0, 2.0, 3.0, 10.0, 20.0, 30.0]);
    }

    #[test]
    fn sends_to_local_listener() {
        let rx = UdpSocket::bind("127.0.0.1:0").unwrap();
        let port = rx.local_addr().unwrap().port();
        let mut s = UdpSender::default();
        s.send("127.0.0.1", port, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let mut buf = [0u8; 64];
        let n = rx.recv(&mut buf).unwrap();
        assert_eq!(n, 48);
        assert_eq!(f64::from_le_bytes(buf[0..8].try_into().unwrap()), 4.0);
    }
}
