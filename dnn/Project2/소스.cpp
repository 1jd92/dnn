#include <stdio.h>
#include <memory.h>
#include <stdlib.h>
#include <time.h>
#include <cmath>
#include <iostream>
using namespace std;

template<typename T, typename U>
class _dnn;
template<typename T, typename U>
class _dnn_node;
template<typename T, typename U>
class _dnn2;
template<typename T>
class dnnlayer;
template<typename T>
class dnnconnection;
template<typename T>
class dnnconnectionlist;
template<typename T>
class dnnfloor;
template<typename T>
class dnn;
template<typename T>
class _math {
private:
	T leaky;
public:
	_math() {
		leaky = static_cast<T>(0);
	}
	inline T _exp(const T a) noexcept {
		return exp(a);
	}
	inline T _log(const T a) noexcept {
		return log(a);
	}
	void setleaky(const T val) noexcept {
		leaky = val;
	}
	T ReLU(const T val) const noexcept {
		if (val > static_cast<T>(0)) return val;
		else if (leaky) return leaky * val;
		else return static_cast<T>(0);
	}
	T dReLU(const T val) const noexcept {
		if (val > static_cast<T>(0)) return static_cast<T>(1);
		else return leaky;
	}
};
template<>
class _math<double> {
private:
	double leaky;
public:
	_math() {
		leaky = 0.;
	}
	double _exp(double a) const noexcept {
		union { double d; long long x; } u;
		u.x = static_cast<long long>(6497320848556798LL * a + 0x3fef127e83d16f12LL);
		return u.d;
	}
	double _log(double a) const noexcept{
		union { double d; long long x; } u = { a };
		return (u.x - 4606921278410026770) * 1.539095918623324e-16;
	}
	void setleaky(const double val) noexcept {
		leaky = val;
	}
	double ReLU(const double val) const noexcept{
		if (val > 0.) return val;
		else if (leaky) return leaky * val;
		else return 0.;
	}
	double dReLU(const double val) const noexcept {
		if (val > 0.) return 1.;
		else return leaky;
	}
};
template<>
class _math<float> {
private:
	float leaky;
public:
	_math() {
		leaky = 0.f;
	}
	float _exp(float a) const noexcept {
		union { float f; int x; } u;
		u.x = (int)(12102203 * a + 1064866805);
		return u.f;
	}
	float _log(float a) const noexcept {
		union { float f; int x; } u = { a };
		return (u.x - 1064866805) * 8.262958405176314e-8f;
	}
	void setleaky(const float val) noexcept {
		leaky = val;
	}
	float ReLU(const float val) const noexcept {
		if (val > 0.f) return val;
		else if (leaky) return leaky * val;
		else return 0.f;
	}
	float dReLU(const float val) const noexcept {
		if (val > 0.) return 1.f;
		else return leaky;
	}
};
template <typename T, typename U>
class _dnn {
protected:
	U **mem, **emem;
	int pmem, nmem;
public:
	_dnn() noexcept {
		pmem = 0;
		nmem = 100;
		mem = new U*[100];
		emem = mem;
	}
	template<typename... V>
	inline void add(V... ar) noexcept {
		static U** mem2;
		if (pmem + 1 >= nmem) {
			mem2 = new U*[nmem + 100];
			memcpy(mem2, mem, sizeof(U*) * nmem);
			nmem += 100;
			delete[] mem;
			mem = mem2;
			emem = mem + pmem;
		}
		*(emem++) = new U(ar...);
		pmem++;
	}
	U& operator[](int n) {
		if (n < 0) {
			if (pmem + n >= 0) return *(mem[pmem + n]);
			else throw(runtime_error("wrong memory access"));
		}
		else {
			if (n < pmem) return *(mem[n]);
			else throw(runtime_error("wrong memory access"));
		}
	}
	virtual ~_dnn() {
		for (U** _mem = mem; _mem != emem; delete _mem++);
		delete[] mem;
	}
};

template<typename T, typename U>
class _dnn_node :public _dnn<T, U> {
public:
	void calculate() noexcept {
		static U **i;
		for (i = _dnn<T, U>::mem; i != _dnn<T, U>::emem; i++) (*i)->calculate();
	}
};



enum _pooltype {
	batchnormalization = 1,
	softmax = 2,
	avgpool = 4,
	maxpool = 5,
	secondpool = 6
};
template <typename T>
class dnnlayer :virtual public _math<T> {
protected:
	mutable int i, j, k, l, m;
	mutable T a, b, c;
	mutable dnnconnection<T> **da, *db;
	mutable dnnlayer<T> *dc;
	dnnconnection<T> **memfrom, **ememfrom, **memto, **ememto;
	int pmemfrom, nmemfrom, pmemto, nmemto;
	T* dat;
	int n, sqrtn;
	T tn, bnavg, bnvar;
	bool isallocated;
	int pooltype;
public:
	dnnlayer(int n, int pooltype = 0) noexcept : n(n), pooltype(pooltype) {
		pmemfrom = 0; nmemfrom = 100; pmemto = 0; nmemto = 100;
		memfrom = new dnnconnection<T> *[100];
		memto = new dnnconnection<T> *[100];
		ememfrom = memfrom; ememto = memto;
		isallocated = false;
		sqrtn = static_cast<int>(sqrt(static_cast<double>(n)));
		if (n - sqrtn * sqrtn) sqrtn++;
		tn = static_cast<T>(n); bnavg = static_cast<T>(0); bnvar = static_cast<T>(1);
	}
	inline void addfrom(dnnconnection<T> &ar) noexcept {
		if (pmemfrom + 1 >= nmemfrom) {
			dnnconnection<T>** memfrom2 = new dnnconnection<T> *[nmemfrom + 100];
			memcpy(memfrom2, memfrom, sizeof(dnnconnection<T>*) * nmemfrom);
			nmemfrom += 100;
			delete[] memfrom;
			memfrom = memfrom2;
		}
		*(ememfrom++) = &ar;
		ar.next = this;
		pmemfrom++;
	}
	inline void addto(dnnconnection<T> &ar) noexcept {
		if (pmemto + 1 >= nmemto) {
			dnnconnection<T>** memto2 = new dnnconnection<T> *[nmemto + 100];
			memcpy(memto2, memto, sizeof(dnnconnection<T>*) * nmemto);
			nmemto += 100;
			delete[] memto;
			memto = memto2;
		}
		*(ememto++) = &ar;
		ar.prev = this;
		pmemto++;
	}
	void loadptr(T* ptr) noexcept {
		if (isallocated) {
			isallocated = 0;
			delete[] dat;
		}
		dat = ptr;
	}
	void allocate() noexcept {
		isallocated = true;
		dat = new T[n];
	}
	inline void calculate_rst() {
		for (i = 0; i < n; i++) dat[i] = static_cast<T>(0.);
	}
	inline void calculate_con() {
		for (da = memfrom; da != ememfrom; da++) {
			db = *da; dc = db->prev; i = dc->n;
			for (j = 0; j < n; j++) {
				for (k = 0; k < i; k++) {
					if ((l = db->nid[k][j]) != -1) dat[j] += db->weight[l] * dc->dat[k];
				}
				dat[j] += db->weight[db->nid[i][j]];
			}
		}
	}
	inline void calculate_bn() {
		a = 0; b = 0; tn = static_cast<T>(n);
		for (i = 0; i < n; i++) a += dat[i];
		a /= tn;
		for (i = 0; i < n; i++) {
			dat[i] -= a;
			b += dat[i] * dat[i];
		}
		b /= tn;
		if (b < static_cast<T>(1e-10)) b = static_cast<T>(1e-10);
		c = bnvar / b;
		for (i = 0; i < n; i++) dat[i] = c * dat[i] + bnavg;
	}
	inline void calculate_smax() {
		a = 0;
		for (i = 0; i < n; i++) {
			dat[i] = _math<T>::_exp(dat[i]);
			a += dat[i];
		}
		for (i = 0; i < n; i++) dat[i] /= a;
	}
	inline void calculate_avgp() {
		i = 0;
		for (da = memfrom; da != ememfrom; da++) {
			db = *da; dc = db->prev; l = dc->n;
			i += db->area*db->area;
			for (j = 0; j < n; j++)
				for (k = 0; k < l; k++)
					if (db->nid[k][j] != -1) dat[j] += dc->dat[k];
		}
		for (j = 0; j < n; j++) dat[j] /= i;
	}
	inline void calculate_maxp() {
		l = 0;
		for (i = 0; i < n; i++) dat[i] = static_cast<T>(-NAN);
		for (da = memfrom; da != ememfrom; da++) {
			db = *da; dc = db->prev; m = dc->n;
			l += db->area*db->area;
			for (j = 0; j < n; j++)
				for (k = 0; k < m; k++)
					if (db->nid[k][j] != -1) {
						if (dat[j] > dc->dat[k]);
						else dat[j] = dc->dat[k];
					}
		}
	}
	inline void calculate_2ndp() {
		l = 0;
		for (i = 0; i < n; i++) dat[i] = static_cast<T>(-NAN);
		for (da = memfrom; da != ememfrom; da++) {
			db = *da; dc = db->prev; m = dc->n;
			l += db->area*db->area;
			for (j = 0; j < n; j++)
				for (k = 0; k < m; k++)
					if (db->nid[k][j] != -1) {
						if (dat[j] > dc->dat[k]);
						else dat[j] = dc->dat[k];
					}
		}
	}
	inline void calculate_ReLU() {
		for (i = 0; i < n; i++) dat[i] = _math<T>::ReLU(dat[i]);
	}
	void calculate() {
		if (pooltype < 5) {
			calculate_rst();
			if (pooltype == 4) calculate_avgp();
			else {
				calculate_con();
				if (pooltype & 1) calculate_bn();
				if (pooltype > 1) calculate_smax();
				calculate_ReLU();
			}
		}
		else if (pooltype == 5) calculate_maxp();
		else calculate_2ndp();
	}
	virtual ~dnnlayer() {
		if (isallocated) delete[] dat;
		for (dnnconnection<T>** _memfrom = memfrom; _memfrom != ememfrom; delete _memfrom++);
		delete[] memfrom;
		for (dnnconnection<T>** _memto = memto; _memto != ememto; delete _memto++);
		delete[] memto;
	}
	friend class dnnconnection<T>;
	friend class dnn<T>;
};

template <typename T>
class dnnfloor : public _dnn_node<T, dnnlayer<T>> {
public:
	friend class dnn<T>;
};
template <typename T>
class dnn : virtual public _dnn_node <T, dnnfloor<T>>,_math<T> {
	mutable int i, j, k;
	mutable T a,e;
	mutable dnnfloor<T>* b;
	mutable dnnlayer<T>* c;
	mutable T *d;
	mutable dnnconnection<T> **da, **db;
	mutable dnnfloor<T> **dc, **dd;
	mutable dnnlayer<T> **de, **df;
	T *_ans;
	T *_ans_mem;
	T *_out;
	int nout;
	T **_in;
	int nin;
	T error;
	dnnlayer<T>* lin;
	dnnlayer<T>* lout;
	dnnconnectionlist<T> cl;
public:
	dnn() {
		nout = -1;_out = nullptr; lout = nullptr; _ans_mem = nullptr; nin = -1; _in = nullptr; lin = nullptr;
	}
	void allocate() {
		for (da = cl._dnn <T, dnnconnection<T>>::mem, db = cl._dnn <T, dnnconnection<T>>::emem; da != db; da++) (*da)->allocate();
		i = 0;
		for (dc = _dnn <T, dnnfloor<T>>::mem,dd= _dnn <T, dnnfloor<T>>::emem; dc != dd; dc++)
			for (de = (*dc)->_dnn <T, dnnlayer<T>>::mem, df = (*dc)->_dnn <T, dnnlayer<T>>::emem; de != df; de++)
				if (i) (*de)->allocate();
				else i = 1;
		if (_dnn <T, dnnfloor<T>>::pmem) {
			b = _dnn <T, dnnfloor<T>>::mem[_dnn <T, dnnfloor<T>>::pmem - 1];
			if (b->_dnn <T, dnnlayer<T>>::pmem) {
				nout = b->_dnn<T, dnnlayer<T>>::mem[0]->n;
				_out = b->_dnn<T, dnnlayer<T>>::mem[0]->dat;
				lout = b->_dnn<T, dnnlayer<T>>::mem[0];
				_ans_mem = new T[nout];
			}
			else throw(runtime_error("wrong memory access"));
			b = _dnn <T, dnnfloor<T>>::mem[0];
			if (b->_dnn <T, dnnlayer<T>>::pmem) {
				nin = b->_dnn<T, dnnlayer<T>>::mem[0]->n;
				_in=&(b->_dnn<T, dnnlayer<T>>::mem[0]->dat);
				lin = b->_dnn<T, dnnlayer<T>>::mem[0];
			}
			else throw(runtime_error("wrong memory access"));
		}
		else throw(runtime_error("wrong memory access"));
	}

	dnnconnectionlist<T>& connection() {
		return cl;
	}
	void setinput(T* in) {
		if (lin) lin->loadptr(in);
		else throw(runtime_error("wrong memory access"));
	}
	void setoutput(T* out) {
		_ans = out;
	}
	void setanswer(int n) {
		if (nout) {
			_ans = _ans_mem;
			for (i = 0; i < nout; i++) _ans[i] = static_cast<T>(0);
			_ans[n] = static_cast<T>(1);
		}
		else throw (runtime_error("wrong memory access"));
	}
	T* getoutput() {
		if (lout) return _out;
		else throw (runtime_error("wrong memory access"));
	}
	T feedforward() {
		if (lout){
			_dnn_node<T, dnnfloor<T>>::calculate();
			a = static_cast<T>(0);
			for (i = 0; i < nout; i++) {
				if ((e = _out[i]) < static_cast<T>(1e-10)) e = static_cast<T>(1e-10);
				a -= _ans[i] * _math<T>::_log(e);
			}
			error = a;
			return a;
		}
		else throw (runtime_error("wrong memory access"));
	}
	int getpredict() {
		if (lout) {
			j = -1; a = static_cast<T>(-NAN);
			for (i = 0; i < nout; i++) {
				if (a > _out[i]);
				else {
					a = _out[i], j = i;
				}
			}
		}
		return j;
	}
	int getpredict(T &probablity) {
		probablity = a;
		return getpredict();
	}
	int getpredict(T &probablity, T &errorval) {
		probablity = a;
		errorval = error;
		return getpredict();
	}
	void backpropagation(T* out) {

	}
};

template <typename T>
class dnnconnection {
public:
	dnnlayer<T> *prev = 0, *next = 0;
	T* weight;
	int* weight_dropout;
	int nweight;
	int nweight_withoutbias;
	int freedom;
	int area;//input size
	double alpha, momentum;
	int dropout;
	int nfrom, nto;
	int rnfrom, rnto;
	int** nid;
public:
	dnnconnection(dnnlayer<T> &prev, dnnlayer<T> &next, int freedom, int area, double alpha, double momentum, double dropout)
		:freedom(freedom), area(area), alpha(alpha), momentum(momentum), dropout(static_cast<int>(dropout*32768.)) {
		prev.addto(*this);
		next.addfrom(*this);
	}
	dnnconnection(int freedom, int area, double alpha, double momentum, double dropout)
		:freedom(freedom), area(area), alpha(alpha), momentum(momentum), dropout(static_cast<int>(dropout*32768.)) {
		prev = nullptr;
		next = nullptr;
	}
	void allocate() {
		static int i, j, k, l, t, u, upd;
		nto = prev->n;
		rnto = prev->sqrtn;
		nfrom = next->n;
		rnfrom = next->sqrtn;
		if (freedom < 0) freedom = nfrom / (-freedom);
		if (area < 0) area = rnto * 2 / (-area);
		nweight = 0;
		if (next->pooltype>3) {
			nid = new int*[nto + 1];
			for (i = 0; i < nto + 1; i++) {
				nid[i] = new int[nfrom];
				for (j = 0; j < nfrom; j++) nid[i][j] = -1;
			}
			for (k = 0; k < area; k++)
				for (l = 0; l < area; l++)
					for (i = 0; i < freedom; i++)
						for (j = i; j < nfrom; j += freedom) {
							t = k - area / 2 + (j % rnfrom) * rnto / rnfrom;
							u = l - area / 2 + (j / rnfrom) * rnto / rnfrom;
							if (t >= 0 && t < rnto && u >= 0 && u < rnto)
								if (t * rnto + u < nto) nid[t * rnto + u][j] = 0;
						}
			nweight_withoutbias = 0;
			weight = nullptr; weight_dropout = nullptr;
		}
		else {
			nid = new int*[nto + 1];
			for (i = 0; i < nto + 1; i++) {
				nid[i] = new int[nfrom];
				for (j = 0; j < nfrom; j++) nid[i][j] = -1;
			}
			upd = 0;
			for (k = 0; k < area; k++)
				for (l = 0; l < area; l++)
					for (i = 0; i < freedom; i++) {
						for (j = i; j < nfrom; j += freedom) {
							t = k - area / 2 + (j % rnfrom) * rnto / rnfrom;
							u = l - area / 2 + (j / rnfrom) * rnto / rnfrom;
							if (t >= 0 && t < rnto && u >= 0 && u < rnto)
								if (t * rnto + u < nto) {
									upd = 1;
									nid[t * rnto + u][j] = nweight;
								}
						}
						if (upd) upd = 0, nweight++;
					}
			nweight_withoutbias = nweight;
			for (i = 0; i < freedom; i++) {
				for (j = i; j < nfrom; j += freedom) nid[nto][j] = nweight;
				nweight++;
			}
			weight = new T[nweight];
			weight_dropout = new int[nweight];
			for (i = 0; i < nweight_withoutbias; i++) {
				weight[i] = static_cast<T>(1.) / (static_cast<T>(area) * static_cast<T>(area));
				weight_dropout[i] = 1;
			}
			for (i = nweight_withoutbias; i < nweight; i++) {
				weight[i] = 0;
				weight_dropout[i] = 1;
			}
		}
	}
	void padding(T val, T bias) {
		for (int i = 0; i < nweight_withoutbias; i++) weight[i] = val;
		for (int i = nweight_withoutbias; i < nweight; i++) weight[i] = bias;
	}
	void _dropout() {
		for (int i = 0; i < nweight; i++) weight_dropout[i] = ((rand() & 32767) < dropout);
	}
	~dnnconnection() {
		if (nid) {
			for (int i = 0; i < nfrom; i++) delete[] nid[i];
			delete[] nid;
		}
		if (weight) delete[] weight;
		if (weight_dropout) delete[] weight_dropout;
	}
	friend class dnnlayer<T>;
	friend class dnn<T>;
};
template <typename T>
class dnnconnectionlist : public _dnn <T, dnnconnection<T>> {
public:
	friend class dnn<T>;
};
int main() {
	srand(time(0));
	try {
		float in[3] = { 0.1f,0.1f,0.1f };
		dnn<float> a;
		a.add();
		a.add();
		a[0].add(3);
		a[1].add(5); 
		a.connection().add(a[0][0], a[1][0], -1, -1, 0, 1, 1);
		a.allocate();
		a.feedforward();

		cout << a.connection()[-1].nweight << " " << a.connection()[-1].nweight_withoutbias;
	}
	catch (runtime_error a) {
		cout << a.what() << endl;
	}
	/*
	try {
		dnn<float> a;
		//double alpha,double momentum,double dropout
		a.add(0.1, 0, 0);
		a.add(0.1, 0, 0);
		a.add(0.1, 0, 0);
		a.add(0.1, 0, 0);
		a.add(0.1, 0, 0);
		a.add(0.1, 0, 0);
		a.add(0.1, 0, 0);
		a.add(0.1, 0, 0);
		//int n
		a[0].add(28 * 28);
		for (int i = 0; i < 32; i++) a[1].add(28 * 28);
		for (int i = 0; i < 32; i++) a[2].add(14 * 14);
		for (int i = 0; i < 64; i++) a[3].add(14 * 14);
		for (int i = 0; i < 64; i++) a[4].add(7 * 7);
		a[5].add(3136);
		a[6].add(128);
		a[7].add(10);
		for (int i = 0; i < 32; i++) {
			//int freedom, int area, int ispool, int pooltype
			a.connection().add(1, 3, 0, 1);
			a[0][0].addfrom(a.connection()[-1]);
			a[1][i].addto(a.connection()[-1]);
		}
		for (int i = 0; i < 32; i++) {
			for (int j = 0; j < 32; j++) {
				a.connection().add(1, 2, 1, 1);
				a[1][i].addfrom(a.connection()[-1]);
				a[2][j].addto(a.connection()[-1]);
			}
		}
		for (int i = 0; i < 32; i++) {
			for (int j = 0; j < 64; j++) {
				a.connection().add(1, 3, 0, 1);
				a[2][i].addfrom(a.connection()[-1]);
				a[3][j].addto(a.connection()[-1]);
			}
		}
		for (int i = 0; i < 64; i++) {
			for (int j = 0; j < 64; j++) {
				a.connection().add(1, 2, 1, 1);
				a[3][i].addfrom(a.connection()[-1]);
				a[4][j].addto(a.connection()[-1]);
			}
		}
		for (int i = 0; i < 64; i++) {
			a.connection().add(-1, -1, 0, 1);
			a[4][i].addfrom(a.connection()[-1]);
			a[5][0].addto(a.connection()[-1]);
		}
		a.connection().add(-1, -1, 0, 1);
		a[5][0].addfrom(a.connection()[-1]);
		a[6][0].addto(a.connection()[-1]);

		a.connection().add(-1, -1, 0, 1);
		a[6][0].addfrom(a.connection()[-1]);
		a[7][0].addto(a.connection()[-1]);
	}
	catch (runtime_error a) {
		cout << a.what() << endl;
	}
	*/
}
