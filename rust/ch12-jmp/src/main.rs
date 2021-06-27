#![feature(link_llvm_intrinsics)]
#![allow(non_camel_case_types)]
#![cfg(not(windows))]

extern crate libc;

use libc::{SIGHUP, SIGUSR1, SIGALRM, SIGQUIT, SIGTERM};
use std::mem;

const JMP_BUF_WIDTH: usize = mem::size_of::<usize>() * 8;
type jmp_buf = [i8; JMP_BUF_WIDTH];

static mut SHUT_DOWN: bool = false;
static mut RETURN_HERE: jmp_buf = [0; JMP_BUF_WIDTH];
const MOCK_SIGNAL_AT: usize = 3;

extern "C" {
    #[link_name = "llvm.setjmp"]
    pub fn setjmp(a: *mut i8) -> i32;

    #[link_name = "llvm.longjmp"]
    pub fn longjmp(a: *mut i8, b: i32) -> ();
}

#[inline]
fn ptr_to_jmp_buf() -> *mut i8 {
    unsafe {
        &RETURN_HERE as *const i8 as *mut i8
    }
}

#[inline]
fn return_early() {
    let franken_pointer = ptr_to_jmp_buf();
    unsafe {
        longjmp(franken_pointer, 1)
    };
}

fn register_signal_handler() {
    unsafe {
        libc::signal(SIGUSR1, handle_signals as usize);
    }
}

#[allow(dead_code)]
fn handle_signals(_signal: i32) {
    register_signal_handler();

    unsafe {
        SHUT_DOWN = true;
    }

    return_early();
}

fn print_depth(depth: usize) {
    for _ in 0..depth {
        print!("#");
    }
    println!("");
}

fn dive(depth: usize, max_depth: usize) {
    unsafe {
        if SHUT_DOWN == true {
            println!("!");
            return;
        }
    }
    print_depth(depth);

    if depth >= max_depth {
        return;
    } else if depth == MOCK_SIGNAL_AT {
        unsafe {
            libc::raise(SIGUSR1);
        }
    } else {
        dive(depth + 1, max_depth);
    }
    print_depth(depth);
}

fn main() {
    const JUMP_SET: i32 = 0;

    register_signal_handler();

    let return_point = ptr_to_jmp_buf();
    let rc = unsafe {
        setjmp(return_point);
    };

    if rc == JUMP_SET {
        dive(0, 10);
    } else {
        println!("early return!");
    }

    println!("finishing!")
}
