<code inline="">emplace_back</code> 和 <code inline="">push_back</code> 是 C++ 容器（比如 <code inline="">std::vector</code>、<code inline="">std::list</code> 等）里常用的两个插入函数。<br>
它们功能相似，但有本质区别 —— <strong>是否需要“临时对象 + 拷贝/移动”</strong>。</p>
<p>下面我们详细对比👇</p>
<hr>
<h2>🧩 一、基本区别</h2>

对比项 | push_back | emplace_back
-- | -- | --
语义 | “把一个现成的对象放进去” | “在容器内部原地构造一个对象”
参数 | 必须是一个现成对象（或能转换成对象） | 可以直接传入构造参数
是否会产生临时对象 | ✅ 可能会（拷贝或移动） | ❌ 不会（直接构造）
性能 | 稍慢 | 稍快（省一次拷贝或移动）
出现版本 | C++98 | C++11 引入


<hr>
<h2>🧠 二、举例理解</h2>
<p>假设我们有一个类：</p>
<pre><code class="language-cpp">struct Worker {
    Worker(int id, std::string name) {
        std::cout &lt;&lt; "construct Worker(" &lt;&lt; id &lt;&lt; ", " &lt;&lt; name &lt;&lt; ")\n";
    }
};
</code></pre>
<p>我们想把它放进 <code inline="">std::vector&lt;Worker&gt; workers;</code></p>
<hr>
<h3>✳️ 用 <code inline="">push_back</code></h3>
<pre><code class="language-cpp">workers.push_back(Worker(1, "Alice"));
</code></pre>
<p>执行过程：</p>
<ol>
<li>
<p>先构造一个临时对象 <code inline="">Worker(1, "Alice")</code></p>
</li>
<li>
<p>再<strong>移动或拷贝</strong>进 <code inline="">vector</code></p>
</li>
<li>
<p>最后销毁临时对象</p>
</li>
</ol>
<p>🧾 输出：</p>
<pre><code>construct Worker(1, Alice)
(move or copy constructor called)
</code></pre>
<hr>
<h3>✳️ 用 <code inline="">emplace_back</code></h3>
<pre><code class="language-cpp">workers.emplace_back(1, "Alice");
</code></pre>
<p>执行过程：</p>
<ol>
<li>
<p><strong>直接在容器内部构造</strong>这个 <code inline="">Worker</code> 对象</p>
</li>
<li>
<p>不会产生临时对象、也没有移动/拷贝</p>
</li>
</ol>
<p>🧾 输出：</p>
<pre><code>construct Worker(1, Alice)
</code></pre>
<hr>
<h2>🧩 三、在你的线程池中</h2>
<p>你的代码：</p>
<pre><code class="language-cpp">workers_.emplace_back([this]{
    ...
});
</code></pre>
<p>这里：</p>
<ul>
<li>
<p><code inline="">workers_</code> 是 <code inline="">std::vector&lt;std::thread&gt;</code></p>
</li>
<li>
<p><code inline="">emplace_back</code> 会直接在 vector 里<strong>原地创建线程对象</strong></p>
</li>
<li>
<p>如果用 <code inline="">push_back</code>，则需要先创建一个临时线程对象，再移动进去：</p>
</li>
</ul>
<pre><code class="language-cpp">workers_.push_back(std::thread([this]{ ... }));  // 多一次移动
</code></pre>
<p>所以：<br>
✅ <code inline="">emplace_back</code> 更简洁、高效、推荐使用。</p>
<hr>
<h2>✅ 四、总结一句话</h2>
<blockquote>
<p><code inline="">push_back</code> 是 “放一个已经建好的对象进去”；<br>
<code inline="">emplace_back</code> 是 “直接在容器里建这个对象”。</p>
</blockquote>