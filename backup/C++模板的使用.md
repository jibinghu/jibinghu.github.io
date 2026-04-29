模板是C++支持[参数化](https://so.csdn.net/so/search?q=%E5%8F%82%E6%95%B0%E5%8C%96&spm=1001.2101.3001.7020)多态的工具，模板的参数有三种类型：类型参数、非类型参数和模板类型参数。
---
用几个例子说明各个模板参数的使用方法：
## 类型参数：
由class或者[typename](https://so.csdn.net/so/search?q=typename&spm=1001.2101.3001.7020)标记的参数，称为类型参数。类型参数是使用模板的主要目的。
- 示例一：
``` cpp
#include <iostream>
#include <iomanip>
#include <string.h>
#include <stdio.h>
 
using namespace std;
 
template <class U,typename V=int>
void add(U &u,V &v){
    cout << u + v<< endl;
}
 
int main()
{
    int a=1,b=2;
    // 调用函数时指定非默认参数类型
    add<int>(a,b);
    double c = 0.5;
    // 覆盖默认参数类型 
    add<int,double>(a,c);
 
    std::cout << "--end--" << endl;
    return 0;
}
```
输出：
``` bash
3
1.5
--end--
```
- 示例二：
``` cpp
#include <iostream>
#include <iomanip>
#include <string.h>
#include <stdio.h>
 
using namespace std;
 
template <class U,typename V=int>
class myclass{
public:
    void add(U &u,V &v){
        cout << u + v<< endl;
    }
};
 
int main()
{
    // 在实例化类时指定参数类型
    myclass<int> m;
    int a = 5,b = 2;
    m.add(a,b);
    // 覆盖默认参数类型 
    myclass<int,double> m2;
    double d = 0.5;
    m2.add(a,d);
 
    std::cout << "--end--" << endl;
    return 0;
}
```
输出：
``` bash
7
5.5
--end--
```
## 非类型参数
非类型参数是指内置类型参数。
定义：
``` cpp
template<typename T, int a> 
class A 
{ 
}; 
```
> 上述代码中，int a就是非类型的模板参数，非类型模板参数为函数模板或类模板预定义一些常量，在模板实例化时，也要求实参必须是常量，即确切的数据值。需要注意的是，非类型参数只能是整型、字符型或枚举、指针、引用类型。非类型参数在所有实例中都具有相同的值，而类型参数在不同的实例中具有不同的值。
示例一：
``` cpp
#include <iostream>
#include <iomanip>
#include <string.h>
#include <stdio.h>
 
using namespace std;
template <typename T,unsigned int len>
class MyString
{
private:
    T array[len];
public:
    // 构造函数
    MyString(){
        for(unsigned int i = 0;i < len;i++){
            this->array[i] = i + 1;
        }
    }
    // 对操作符 [ ] 重载
    T& operator[](unsigned int i){
        if(i >= len)return array[0];
        return this->array[i];
    }
};
 
int main()
{
    // 即类型模板参数需要指定参数类型，而非类型模板参数则需要指定具体数值
    MyString<unsigned int,5> ms1;
    cout << ms1[2] << endl;
 
    MyString<double,5> ms2;
    cout << ms1[3] << endl;
    std::cout << "--end--" << endl;
    return 0;
}
```
> 使用非类型参数时，有以下几点需要注意。
（1）调用非类型参数的实参必须是常量表达式，即必须能在编译时计算出结果。
（2）任何局部对象、局部变量的地址都不是常量表达式，不能用作非类型的实参，全局指针类型、全局变量也不是常量表达式，也不能用作非类型的实参。
（3）sizeof()表达式结果是一个常量表达式，可以用作非类型的实参。
（4）非类型参数一般不用于函数模板。
## 模板类型参数：
> 模板类型参数就是模板的参数为另一个模板。
定义：
``` cpp
template<typename T, template<typename U, typename Z> class A> 
class Parameter 
{ 
    A<T,T> a; 
};
```
 上述代码中，类模板Param eter的第二个模板参数就是一个类模板。需要注意的是，只有类模板可以作为模板参数，参数声明中必须要有关键字class。
``` cpp
// 例子确实写的比较绕，知道怎么回事就可以了
#include <iostream>
#include <iomanip>
#include <string.h>
#include <stdio.h>
using namespace std;
 
template <class T,class U>
class myclass
{
public:
    T _t;
    U _u;
    myclass(){
        cout << "myclass init 1" << endl;
    }
    myclass(T &t,U &u):_t(t),_u(u){
        cout << "myclass init 2" << endl;
    }
    ~myclass(){
        cout << "myclass delete" << endl;
    }
};
template <typename T,template<typename U,typename Z> class myclass>
class OtherClass{
public:
    // 引用构造实例化 _m 对象
    myclass<T,T> _m;
    OtherClass(myclass<T,T> &m):_m(m){
    }
    void show(){
        cout << _m._t << endl;
        cout << _m._u << endl;
    }
};
 
int main()
{
    int a = 5;
    int b = 6;
    myclass<int,int> m(a,b);
    printf("after myclass!\n");
    OtherClass<int,myclass> o(m);
    printf("after otherclass!\n");
    o.show();
 
    std::cout << "--end--" << endl;
    return 0;
}
```
输出：
``` bash
myclass init 2
after myclass!
after otherclass!
5
6
--end--
// 这里我没明白为什么会调用一次构造函数和两次析构函数，等下来补全说明
myclass delete
myclass delete
```



