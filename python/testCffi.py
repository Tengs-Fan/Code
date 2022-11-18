from cffi import FFI
ffi = FFI()
ffi.set_source("_test", """
long factorial(int n) {
    long r = n;
    while(n > 1) {
        n -= 1;
        r *= n;
    }
    return r;
}
""")
ffi.cdef("""long factorial(int);""")
ffi.compile()
from _test import lib     # import the compiled library
print(lib.factorial(100))  # 3628800
