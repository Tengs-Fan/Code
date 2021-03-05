fn main()
{
//    println!("Hello World")
//	println!("10 + 10 = {}", 10 + 10);
//	let name = "Brad";
//	let age = 37;
//	println!("My name is {} and I am {}", name, age);)
	

	// 打印tuple，注意下标访问方式不同于寻常
	let person: (&str, &str, i8) = ("Brad", "Mass", 37);
	println!("{} is from {} and is {}", person.0, person.1, person.2);

	// 在{}中使用一定规则就可以达到想要的输出，不需要额外函数对10进制数字进行转换
	println!("Binary: {:b} Hex: {:x} Octal: {:o}", 10, 10, 10);

	// (12, true, "hello")是一个tuple，不可以直接输出，因此要使用{:?}debug trait
	println!("{:?}", (12, true, "hello"));





}
