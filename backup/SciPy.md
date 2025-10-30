<h2>🧠 一、SciPy 是什么？</h2>
<p><strong>SciPy</strong>（读作 “Sai-pai”）是 Python 的一个<strong>科学计算库</strong>，全名是 <strong>Scientific Python</strong>。<br>
它建立在 <strong>NumPy</strong>（数值计算的基础库）之上，是专门为 <strong>科学、工程、数学计算</strong> 提供高级工具的集合。</p>
<p>简而言之：</p>
<blockquote>
<p>🔹 <strong>NumPy</strong> 负责高效的数值运算（数组、矩阵、线性代数）<br>
🔹 <strong>SciPy</strong> 在此基础上提供更高层次的数学与科学算法（积分、微分方程、优化、插值、统计等）</p>
</blockquote>
<hr>
<h2>🧩 二、SciPy 的主要模块</h2>
<p>SciPy 包含很多子模块，每个模块负责不同方向的科学计算任务。</p>

模块名 | 主要功能 | 示例
-- | -- | --
scipy.integrate | 积分、求解微分方程 | quad()、solve_ivp()
scipy.optimize | 优化、最小化问题 | minimize()、curve_fit()
scipy.interpolate | 插值与平滑 | interp1d()、griddata()
scipy.fft | 快速傅里叶变换 | fft()、ifft()
scipy.linalg | 线性代数运算（比 numpy 更强） | inv()、eig()
scipy.spatial | 空间数据结构与距离计算 | KDTree、distance_matrix()
scipy.stats | 概率统计分析 | norm.pdf()、ttest_ind()
scipy.signal | 信号处理 | convolve()、find_peaks()
scipy.ndimage | 多维图像处理 | gaussian_filter()、sobel()


<p>SciPy 是 NumPy 的“进阶扩展”。<br>
大部分 SciPy 函数都以 NumPy 数组为输入输出。</p>
<hr>
<h2>🧩 五、SciPy 的安装</h2>
<pre><code class="language-bash">pip install scipy
</code></pre>
<p>安装后可以这样导入：</p>
<pre><code class="language-python">import scipy
from scipy import integrate, optimize
</code></pre>
<hr>
<h2>🌍 六、应用领域</h2>
<p>SciPy 广泛应用于：</p>
<ul>
<li>
<p>🌡️ 物理建模（热传导、振动方程）</p>
</li>
<li>
<p>📈 金融数学（最优投资、蒙特卡洛）</p>
</li>
<li>
<p>🧬 生物统计与信号处理</p>
</li>
<li>
<p>🤖 机器学习算法原型</p>
</li>
<li>
<p>🌪️ 气象、天体模拟（如 FV3 动力核中的数值积分）</p>
</li>
</ul>
<hr>