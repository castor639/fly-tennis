// Deterministic capture of the Fly Tennis page: steps the page frame by frame in
// headless Chromium (software GL) and writes PNGs, then ffmpeg makes the video/gif.
// usage: node capture.js <page.html> <outdir> <seconds> [fps] [width] [height]
const { chromium } = require('playwright');
const path = require('path'), fs = require('fs');
(async () => {
  const [page_, outdir, secs, fps = '30', W = '1600', H = '900'] = process.argv.slice(2);
  fs.mkdirSync(outdir, { recursive: true });
  const browser = await chromium.launch({ args: ['--use-gl=angle', '--use-angle=swiftshader', '--enable-unsafe-swiftshader', '--ignore-gpu-blocklist', '--enable-webgl', '--no-sandbox'] });
  const page = await browser.newPage({ viewport: { width: +W, height: +H }, deviceScaleFactor: 1 });
  page.on('console', m => { if (m.type() === 'error') console.log('console:', m.text()); });
  await page.goto('file://' + path.resolve(page_) + '?record=1', { waitUntil: 'load' });
  await page.waitForFunction(() => window.__ready === true, null, { timeout: 60000 });
  await page.waitForTimeout(1500);                                   // fonts, env map
  const n = Math.round(+secs * +fps), ticksPer = Math.max(1, Math.round(60 / +fps));
  const t0 = Date.now();
  for (let i = 0; i < n; i++) {
    await page.evaluate(k => window.__step(k), ticksPer);
    await page.screenshot({ path: path.join(outdir, `f${String(i).padStart(4, '0')}.png`), type: 'png' });
    if (i % 50 === 0) console.log(`frame ${i}/${n}  ${((Date.now() - t0) / 1000).toFixed(0)}s`);
  }
  await browser.close();
  console.log('done', n, 'frames');
})();
