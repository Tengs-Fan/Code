#![cfg(not(windows))]

extern crate libc;

use std::time;
use std::thread::{sleep};
use libc::{SIGUSR1, SIGTERM};

static mut SHUT_DOWN: bool = false;

fn main() {
    register_signal_handlers();

    let delay = time::Duration::from_secs(1);

    for i in 1_usize.. {
        println!("{}", i);
        unsafe {
            if SHUT_DOWN {
                println!("*");
                return;
            }
        }

        sleep(delay);

        let signal = if i > 2 {
            SIGTERM
        } else {
            SIGUSR1
        };

        unsafe {
            libc::raise(signal);
        }
    }
    
    unreachable!();
}

fn register_signal_handlers() {
    unsafe {
        libc::signal(SIGTERM, handle_sigterm as usize);
        libc::signal(SIGUSR1, handle_sigusr1 as usize);
    }
}

#[allow(dead_code)]
fn handle_sigterm(_signal: i32) {
    register_signal_handlers();

    println!("SIGTERM");

    unsafe {
        SHUT_DOWN = true;
    }
}

#[allow(dead_code)]
fn handle_sigusr1(_signal: i32) {
    register_signal_handlers();

    println!("SIGUSR1");

    unsafe {
        SHUT_DOWN = true;
    }
}
