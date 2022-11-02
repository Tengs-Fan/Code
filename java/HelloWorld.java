import java.io.*;

class HelloWorld{
	public static void main(String args[]){
		int i = 0x12345678;
		byte b = (byte) (i >>>6);
		System.out.printf("%x\n",b); // -22
	}
}
