use std::{thread, time};

fn add (a: i32, b: i32) -> i32 {
    a + b
}

fn main() {
    let lambda_add = |a,b| {a+b};
    assert_eq!(add(4,5), lambda_add(4,5));

    let start = time::Instant::now();
    
    let handler1 = thread::spawn(move || {
        let pause = time::Duration::from_millis(300);
        thread::sleep(pause.clone());
    });

    let handler2 = thread::spawn(move || {
        let pause = time::Duration::from_millis(300);
        thread::sleep(pause.clone());
    });

    handler1.join().unwrap();
    handler2.join().unwrap();

    let finish = time::Instant::now();

    println!("{:02?}", finish.duration_since(start));
}
