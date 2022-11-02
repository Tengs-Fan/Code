#ifndef	__VECTOR_H_
#define	__VECTOR_H_

#define DEBUG(x) do { std::cerr << x; } while (0)

#include <algorithm>
#include <iostream>

template <typename Object>
class Vector
{
public:
	explicit Vector(int initSize = 0)
		: theSize ( initSize ), theCapacity ( initSize + SPARE_CAPACITY )
	{ objects = new Object[ theCapacity ];
		DEBUG("explicit Vector");
	}

	Vector(const Vector & rhs)
		: theSize ( rhs.theSize ), theCapacity(rhs.theCapacity), objects (nullptr)
	{	
		objects = new Object[ theCapacity ];
		for (int k=0; k<theSize; ++k){ objects[k] = rhs.objects[k];}
		DEBUG("Vector(const Vector & rhs)");
	}

	Vector & operator= (const Vector & rhs) {
		Vector copy = rhs;
		std::swap(*this, copy);
		DEBUG("Vector(Vector & operator= (const Vector & rhs))");
		return *this;
	}

	~Vector()
	{ delete[] objects;
		DEBUG("~Vector()");
	}

	Vector(Vector && rhs)
		: theSize(rhs.theSize), theCapacity(rhs.theCapacity), objects(rhs.objects)
	{
		rhs.objects = nullptr;
		rhs.theSize = 0;
		rhs.theCapacity = 0;
		DEBUG("vector(vector && rhs)");
	}

	Vector & operator= (Vector && rhs) {
		std::swap(theSize, rhs.theSize);
		std::swap(theCapacity, rhs.theCapacity);
		std::swap(objects, rhs.objects);
		DEBUG("Vector & operator= (Vector && rhs)");
		return *this;
	}

	Object & operator[](int index) {
		return objects[index]; }
	const Object & operator[](int index) const {
		return objects[index]; }

	bool empty() const {
		return size() == 0; }
	int size() const {
		return theSize; }
	int capacity() const {
		return theCapacity; }

	void push_back(const Object & x){
		if (theSize == theCapacity) {
			reserve(2*theCapacity + 1); }
		objects[theSize++] = x;
	}

	void push_back(Object && x){
		if (theSize == theCapacity) {
			reserve(2*theCapacity + 1); }
		objects[theSize++] = std::move(x);
	}

	void pop_back() {
		--theSize;
	}

	const Object & back() const {
		return objects[theSize - 1];
	}

	typedef Object* iterator;
	typedef const Object* const_iterator;

	iterator begin() {
		return &objects[0];}
	const_iterator begin() const {
		return &objects[0]; }

	void resize(int newSize) {
		if(newSize > theSize) {
			reserve(newSize * 2);
		}
		theSize = newSize;
	}

	void reserve(int newCapacity) {
		if(newCapacity < theSize) {return;}

		Object *newArray = new Object[newCapacity];
		for (int i=0; i<theSize; ++i) {
			newArray[i] = std::move( objects[i]);
		}
		theCapacity = newCapacity;
		std::swap(objects, newArray);
		delete [] newArray;
	}
	
	static const int SPARE_CAPACITY = 2;
private:
	int theSize;
	int theCapacity;
	Object* objects;

};

#endif	//__VECTOR_H_
