import java.io.*;

class Bits {

	public static void main (String args[])	{
		//int a = ((byte) 0xae) << 16; 
		//a = a & 0x00ff0000;
		//System.out.printf("%x\n", a);

		//int b = ((byte) 0xae) << 16; 
		//b = b | 0x00ff0000;
		//System.out.printf("%x\n", b);

		//int c = ((byte) 0xae) << 16;
		//System.out.printf("%x\n", c);

		//int d = (byte) 0xae;
		//System.out.printf("%x\n", d);

		 byte b = (byte) 0xd6;
		 int i = 0xff & (b << 4);
        System.out.printf("0x%x\n", i);
		//   class Main {
		int a = 0xcf44c48c;
        a = ((0xff & a) << 24) | (a >>> 8);
        System.out.printf("0x%x\n", a);
        

	}


}
