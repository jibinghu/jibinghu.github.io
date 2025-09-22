<html><head></head><body><p>在C++中，<code inline="">assert</code>是一个宏，用于在程序运行时进行条件检查。如果条件不满足，程序会输出错误信息并终止执行。<code inline="">assert</code>通常用于调试阶段，帮助开发人员捕捉程序中的潜在错误。</p>
<h3>1. <code inline="">assert</code> 的基本用法</h3>
<p><code inline="">assert</code> 语句的基本语法如下：</p>
<pre><code class="language-cpp">#include &lt;cassert&gt;

assert(expression);
</code></pre>
<ul>
<li>
<p><code inline="">expression</code> 是一个布尔表达式。如果 <code inline="">expression</code> 的值为 <code inline="">false</code>，则 <code inline="">assert</code> 会触发。</p>
</li>
<li>
<p>如果条件成立（即 <code inline="">expression</code> 为 <code inline="">true</code>），<code inline="">assert</code> 不会执行任何操作，程序会继续正常运行。</p>
</li>
<li>
<p>如果条件不成立（即 <code inline="">expression</code> 为 <code inline="">false</code>），<code inline="">assert</code> 会输出一条错误消息，包含：</p>
<ul>
<li>
<p>触发断言的文件名和行号</p>
</li>
<li>
<p><code inline="">expression</code> 的内容</p>
</li>
<li>
<p>然后程序终止。</p>
</li>
</ul>
</li>
</ul>
<p>例如：</p>
<pre><code class="language-cpp">#include &lt;cassert&gt;

int main() {
    int x = -5;
    assert(x &gt;= 0);  // 如果x小于0，程序会打印错误信息并终止
    return 0;
}
</code></pre>
<p>如果 <code inline="">x</code> 小于 <code inline="">0</code>，<code inline="">assert</code> 会输出类似以下的错误信息：</p>
<pre><code>Assertion failed: x &gt;= 0, file example.cpp, line 6
</code></pre>
<h3>2. 静态断言（<code inline="">static_assert</code>）</h3>
<p>静态断言是在编译时进行的断言。与普通的 <code inline="">assert</code> 不同，<code inline="">static_assert</code> 会在编译期间检查条件，而不是在程序运行时。</p>
<p><code inline="">static_assert</code> 用法如下：</p>
<pre><code class="language-cpp">static_assert(expression, "Error message");
</code></pre>
<ul>
<li>
<p><code inline="">expression</code> 必须是一个常量表达式（常量表达式是编译时已知的）。</p>
</li>
<li>
<p>如果条件不成立，编译器会报告错误，显示 "Error message"。</p>
</li>
</ul>
<p>例如：</p>
<pre><code class="language-cpp">static_assert(sizeof(int) == 4, "Integers should be 4 bytes");
</code></pre>
<p>如果 <code inline="">int</code> 类型的大小不是 4 字节，编译器会报错，提示 <code inline="">"Integers should be 4 bytes"</code>。</p>
<p>静态断言常用于确保某些编译时条件被满足，特别是在模板编程中，确保类型大小或特定类型满足预期。</p>
<h3>3. 动态断言（<code inline="">assert</code>）</h3>
<p>动态断言就是我们在第一部分提到的 <code inline="">assert</code>，它是在程序运行时检查条件，而不是编译时。</p>
<ul>
<li>
<p><code inline="">assert</code> 检查的是运行时条件，因此可以用来捕捉那些只有在程序执行过程中才能得知的错误。</p>
</li>
<li>
<p>运行时断言有助于开发人员在调试时发现逻辑错误，避免将这些错误推迟到后续的代码执行。</p>
</li>
</ul>
<h3>4. <code inline="">assert</code> 与 <code inline="">static_assert</code> 的区别</h3>

特性 | assert（动态断言） | static_assert（静态断言）
-- | -- | --
触发时间 | 运行时检查 | 编译时检查
适用条件 | 适用于运行时计算得出的条件 | 适用于常量表达式
错误处理方式 | 程序会输出错误信息并终止程序 | 编译失败，显示错误消息
使用场景 | 调试阶段，捕捉运行时错误 | 编译时约束，确保特定条件满足


<h3>5. <code inline="">assert</code> 在发布版本中的行为</h3>
<p>通常，在发布版本中，C++标准库中的 <code inline="">assert</code> 宏会被禁用。在代码中使用 <code inline="">NDEBUG</code> 宏定义时，<code inline="">assert</code> 会被替换为空语句，因此不会进行任何断言检查。通常在调试版本中启用断言，而在发布版本中禁用断言，以避免影响性能。</p>
<pre><code class="language-cpp">#define NDEBUG
#include &lt;cassert&gt;

int main() {
    assert(1 == 2);  // 这行代码在定义了NDEBUG时不会起作用
    return 0;
}
</code></pre>
<p>定义 <code inline="">NDEBUG</code> 后，<code inline="">assert</code> 宏会被禁用，这使得在生产环境中，断言不会造成性能损失。</p>
<h3>总结</h3>
<ul>
<li>
<p><code inline="">assert</code> 用于运行时检查，帮助开发者在调试阶段发现潜在错误。</p>
</li>
<li>
<p><code inline="">static_assert</code> 用于编译时检查，可以确保类型或大小等特定条件在编译时就被验证。</p>
</li>
<li>
<p><code inline="">assert</code> 在发布版本中通常会被禁用，以提高程序的运行效率。</p>
</li>
</ul>
<p>如果你希望确保某些条件在编译时就被验证，使用 <code inline="">static_assert</code> 是更好的选择。如果你在调试过程中想捕捉运行时的错误，使用 <code inline="">assert</code> 会很有帮助。</p></body></html>