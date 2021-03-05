fn main() {

	//	let pointer: isize = 2333333333333333333333; // pointer size
	//	let _int64: i64 = 2333333333333333333333;    // i64 与 64位机上的 isize 相同
	//	let upointer: usize = 6666666666666666666666; // pointer size
	//	let upointer: u64 = 6666666666666666666666; // u64 与 64位机上的 usize 相同
	//


	// 编译器可以根据上下文推断类型
	// let mut 定义的是可以变的变量，你还不需要知道它的意思
	let mut inferred_type = 12; // Type i64 is inferred from another line
	inferred_type = 4294967296i64;

	// error: 下面这段代码有错误！
	// 你不能这么做不是因为它计算的结果不合法
	// 而是它们类型不相同，并不能进行计算（虽然看上去很明显可以）
//	println!("1u32 - 1i32 = {}", 1u32 - 1i32);
	// 字面量表示了计算机可以直接读懂的数据
	// 这个时候请主要注意等号右边的那些量
	let integer = 1024; // 整数，默认为 i32
	let interger_with_underscore = 1_000_000_000u32; // #1 整数，可以使用下划线分开。如果你不使用默认类型，你需要告诉编译器

	// 整数可以使用不同的进制表示
	let bin = 0b10_000_000_000; // 二进制
	let oct = 0o2_000; // 八进制
	let dec = 1024; // 十进制
	let hex = 0x400; // 十六进制

	// 打印出来默认都是十进制，如果希望有不同的打印方法，则可以查看文档![formatting](https://doc.rust-lang.org/rust-by-example/hello/print/fmt.html)
	println!("four integers: bin: {}, oct: {}, dec: {}, hex: {}", bin, oct, dec, hex);

	let float = 1.0; // 浮点数，默认为 f64
	let float_with_underscore = 0.000_1f32; // 浮点数，可以使用下划线分开。可以通过标注不使用默认类型

	let characters: char = 'a'; // 字符char，使用单引号表示

	let boolean: bool = false; // bool 类型，只有 true 和 false 两个值

	// #2
	// !注：下面这一行没有错误，但是它会阻止程序继续运行
	// 这个语句代表在程序运行中的某个要求，这里要求两个数值相等
	// 这不是一个错误，而是一个运行时的判断
	assert_eq!(true, boolean); // 判断是否相同

	let unit = (); // unit 类型，不占空间，也没有什么意义
	let another_unit = ();
	assert_eq!(unit, another_unit); // 两者是相同的

	let string: &str = "Hello World"; // 字符串 str，可以认为是一个 Unicode的序列。使用双引号表示
	println!("{}!", string);
}
