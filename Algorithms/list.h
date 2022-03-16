#ifndef LIST_H
#define LIST_H

#include <cstdio>

#include <algorithm>
using namespace std;

// Double Linked List
template <typename Object>
class List{
private:
	struct Node{
		Object data;
		Node *prev;
		Node *next;

		Node(const Object &d = Object{}, Node *p = nullptr, Node *n = nullptr)
			: data{d}, prev{p}, next{n}{}
		Node(Object && d, Node *p = nullptr, Node *n = nullptr)
			: data{move(d)}, prev{p}, next{n}{}
	};

private:
	int theSize;
	Node *head;
	Node *tail;

	void init() {
		theSize = 0;
		head = new Node;
		tail = new Node;
		head->next = tail;
		tail->prev = head;
	}

public:
	class const_iterator
	{
	public:
		const_iterator() : current{nullptr}
		{ }

	protected:
		Node *current;
		
	};

	class iterator : public const_iterator {
		iterator()
		{ }
	};

public:
	List() 
	{ init(); }

	int size() const
	{ return theSize; }

	bool empty()
	{ return size() == 0; }

	void clear() {
		while (!empty()) {

		}
	}


	iterator Insert(iterator itr, const Object& x){
		Node *p = itr.current;
		++theSize;
		return iterator( p->prev = p->prev->next = new Node{x, p->prev, p});
	}

	iterator Insert(iterator itr, Object && x) {
		Node  *p = itr.current;
		++theSize;
		return iterator( p->prev = p->prev->next = new Node{std::move(x), p->prev, p});
	}

	iterator erase(iterator itr) {
		Node *p = itr.current;
		iterator retVal (p->next);
		p->prev->next = p->next->next;
		p->next->prev = p->prev->prev;
		delete p;
		--theSize;

		return retVal;
	}

	iterator erase(iterator from, iterator to) {
		for (iterator itr = from; itr != to;) {
			itr = erase(itr);
		}
		return to;
		//to.current->prev = from.current->prev;
		//from.current->next = to.current->next;
		//iterator retVal (to.current->next);
		//theSize --;
		//return retVal;
	}
};

#endif //LIST_H
