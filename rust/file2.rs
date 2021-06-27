#![allow(unused_variables)]

//type File = String;
#[derive(Debug)]
struct File {
    name: String,
    data: Vec<u8>,
}

fn open (f: &mut File) -> bool {
    true
}
 
fn close (f: &mut File) -> bool {
    true
}

fn read(f: &mut File, save_to: &mut Vec<u8>) -> usize {
    let mut tmp = f.data.clone();
    let read_length = tmp.len();
    save_to.reserve(read_length);
    save_to.append(&mut tmp);
    return read_length;
}

fn main() {
    //let mut f1 = File::from("f1.txt");
    let mut f2 = File {
        name: String::from("2.txt"),
        data: vec![114, 117, 115, 116, 33],
    };

    let mut buffer: Vec<u8> = vec![];

    open(&mut f2);
    let f2_length = read(&mut f2, &mut buffer);
    close(&mut f2);

    let text = String::from_utf8_lossy(&buffer);

    println!("{:?}", f2);
    println!("{} is {} bytes long", &f2.name, f2_length);
    println!("{}", text) // <10>  View [114, 117, 115, 116, 33] as an actual word
}
