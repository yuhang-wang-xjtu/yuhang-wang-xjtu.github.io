const { marked } = require('marked');
const fs = require('fs');
const path = require('path');
const katex = require('katex');

const pagesDir = path.join(__dirname, '..', 'pages');
const templatePath = path.join(__dirname, 'blog-template.html');
const outputDir = path.join(__dirname, '..', 'blog');

if (!fs.existsSync(outputDir)) {
  fs.mkdirSync(outputDir, { recursive: true });
}

if (!fs.existsSync(pagesDir)) {
  console.error('pages directory not found');
  process.exit(1);
}

const template = fs.readFileSync(templatePath, 'utf-8');

marked.setOptions({
  breaks: false,
  gfm: true,
});

const posts = [];

function protectMathBlocks(content) {
  const mathBlocks = [];
  let result = content.replace(/\$\$([\s\S]*?)\$\$/g, (match) => {
    mathBlocks.push({ type: 'display', content: match });
    return `\n<!--MATHBLOCK_${mathBlocks.length - 1}-->\n`;
  });
  return { content: result, mathBlocks };
}

function renderMathBlock(block) {
  if (block.type === 'display') {
    const formula = block.content.replace(/^\$\$/, '').replace(/\$\$$/, '').trim();
    try {
      const rendered = katex.renderToString(formula, {
        displayMode: true,
        throwOnError: false,
        strict: 'ignore',
        trust: true,
      });
      return rendered.replace(/<span class="katex-mathml">[\s\S]*?<\/span>/, '');
    } catch (e) {
      return block.content;
    }
  }
  return block.content;
}

function renderInlineMath(html) {
  return html.replace(/\$([^$\n]{1,200}?)\$/g, (match, formula) => {
    if (match.includes('$$')) return match;
    if (/^\s*\d+[\d,\.]*\s*$/.test(formula)) return match;
    try {
      const rendered = katex.renderToString(formula, {
        displayMode: false,
        throwOnError: false,
        strict: 'ignore',
        trust: true,
      });
      return rendered.replace(/<span class="katex-mathml">[\s\S]*?<\/span>/, '');
    } catch (e) {
      return match;
    }
  });
}

function restoreMathBlocks(html, mathBlocks) {
  return html.replace(/<!--MATHBLOCK_(\d+)-->/g, (match, index) => {
    const block = mathBlocks[parseInt(index)];
    if (!block) return match;
    return renderMathBlock(block);
  });
}

function processMarkdown(content) {
  const { content: protectedContent, mathBlocks } = protectMathBlocks(content);
  const html = marked.parse(protectedContent);
  const restored = restoreMathBlocks(html, mathBlocks);
  return renderInlineMath(restored);
}

function applyTemplate(tmpl, vars) {
  let result = tmpl;
  for (const [key, value] of Object.entries(vars)) {
    result = result.split(`{{${key}}}`).join(value);
  }
  return result;
}

function processStandaloneMd(file) {
  const filePath = path.join(pagesDir, file);
  const content = fs.readFileSync(filePath, 'utf-8');
  const slug = file.replace('.md', '');

  let title = slug;
  let description = '';
  const date = new Date().toISOString().slice(0, 10);

  const firstH1 = content.match(/^#\s+(.+)$/m);
  if (firstH1) title = firstH1[1].trim();

  const firstPara = content.match(/^[^#\n].+$/m);
  if (firstPara) description = firstPara[0].trim().substring(0, 160);

  const htmlContent = processMarkdown(content);
  const htmlFileName = slug + '.html';
  const outputPath = path.join(outputDir, htmlFileName);

  let result = applyTemplate(template, {
    TITLE: title,
    DESCRIPTION: description || title,
    DATE: date,
    CONTENT: htmlContent,
    ROOT: '../',
  });

  fs.writeFileSync(outputPath, result, 'utf-8');
  console.log(`  -> blog/${htmlFileName}`);

  posts.push({ title, description, date, slug, url: 'blog/' + htmlFileName });
}

function processFolderPost(folderName) {
  const folderPath = path.join(pagesDir, folderName);
  const mdPath = path.join(folderPath, 'index.md');

  if (!fs.existsSync(mdPath)) return;

  const content = fs.readFileSync(mdPath, 'utf-8');
  const slug = folderName;

  let title = slug;
  let description = '';
  const date = new Date().toISOString().slice(0, 10);

  const firstH1 = content.match(/^#\s+(.+)$/m);
  if (firstH1) title = firstH1[1].trim();

  const firstPara = content.match(/^[^#\n].+$/m);
  if (firstPara) description = firstPara[0].trim().substring(0, 160);

  const postOutputDir = path.join(outputDir, slug);
  if (!fs.existsSync(postOutputDir)) {
    fs.mkdirSync(postOutputDir, { recursive: true });
  }

  const imageFiles = fs.readdirSync(folderPath).filter(f =>
    /\.(png|jpg|jpeg|gif|svg|webp)$/i.test(f)
  );

  imageFiles.forEach(img => {
    const src = path.join(folderPath, img);
    const dest = path.join(postOutputDir, img);
    fs.copyFileSync(src, dest);
  });

  const htmlContent = processMarkdown(content);

  let result = applyTemplate(template, {
    TITLE: title,
    DESCRIPTION: description || title,
    DATE: date,
    CONTENT: htmlContent,
    ROOT: '../../',
  });

  const outputPath = path.join(postOutputDir, 'index.html');
  fs.writeFileSync(outputPath, result, 'utf-8');
  console.log(`  -> blog/${slug}/index.html (+ ${imageFiles.length} images)`);

  posts.push({ title, description, date, slug, url: 'blog/' + slug + '/' });
}

const entries = fs.readdirSync(pagesDir);

entries.forEach(entry => {
  const fullPath = path.join(pagesDir, entry);
  const stat = fs.statSync(fullPath);

  if (stat.isDirectory()) {
    processFolderPost(entry);
  } else if (entry.endsWith('.md')) {
    processStandaloneMd(entry);
  }
});

let blogIndex = generateBlogIndex(posts);
fs.writeFileSync(path.join(outputDir, '..', 'blog.html'), blogIndex, 'utf-8');
console.log('  -> blog.html (listing)');

console.log(`\nBuilt ${posts.length} blog post(s).`);

function generateBlogIndex(posts) {
  const sorted = posts.sort((a, b) => b.date.localeCompare(a.date));

  const cards = sorted.map(p =>
    `<div class="col-md-6 mb-4 fade-up">
          <div class="card blog-card h-100">
            <div class="card-body">
              <div class="card-meta">${p.date} · 博客</div>
              <h5 class="card-title">${p.title}</h5>
              <p class="card-text">${p.description}</p>
              <a href="${p.url}" class="btn btn-outline-primary btn-sm mt-auto">阅读全文 &rarr;</a>
            </div>
          </div>
        </div>`
  ).join('\n');

  return `<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>博客 - 王宇航</title>
  <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0-alpha1/dist/css/bootstrap.min.css" rel="stylesheet"
    integrity="sha384-GLhlTQ8iRABdZLl6O3oVMWSktQOp6b7In1Zl3/Jr59b6EGGoI1aFkw7cmDA6j6gD" crossorigin="anonymous">
  <link href="public/background.css" rel="stylesheet">
  <link href="public/site.css" rel="stylesheet">
</head>
<body>
  <nav class="navbar navbar-expand-lg">
    <div class="container">
      <a class="navbar-brand" href="index.html"><span class="brand-dot"></span>王宇航</a>
      <button class="navbar-toggler" type="button" data-bs-toggle="collapse" data-bs-target="#navbarNav">
        <span class="navbar-toggler-icon"></span>
      </button>
      <div class="collapse navbar-collapse" id="navbarNav">
        <div class="navbar-nav ms-auto">
          <a class="nav-link" href="index.html">首页</a>
          <a class="nav-link" href="About.html">个人介绍</a>
          <a class="nav-link" href="data.html">科研项目</a>
          <a class="nav-link active" href="blog.html">博客</a>
        </div>
      </div>
    </div>
  </nav>

  <div class="container page-section">
    <div class="text-center mb-5">
      <h2 class="section-title">博客</h2>
      <div class="divider"></div>
      <p class="text-muted">论文阅读笔记与技术分享</p>
    </div>
    <div class="row">
${cards}
    </div>
    <div class="text-center mt-3 text-muted">
      <p>更多文章即将更新...</p>
    </div>
  </div>

  <footer class="text-center">
    <div class="container">
      <p class="mb-1">王宇航 · 西安交通大学 · 建筑学 &amp; 计算机科学与技术</p>
      <p class="mb-0 text-muted small">Whale@stu.xjtu.edu.cn</p>
    </div>
  </footer>

  <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0-alpha1/dist/js/bootstrap.bundle.min.js"
    integrity="sha384-/mhDoLbDldZc3qpsJHpLogda//BVZbgYuw6kof4u2FrCedxOtgRZDTHgHUhOCVim"
    crossorigin="anonymous"></script>
  <script src="public/analytics.js"></script>
</body>
</html>`;
}
