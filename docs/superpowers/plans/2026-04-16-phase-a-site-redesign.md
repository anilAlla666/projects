# Phase A — Site Redesign Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the `site/` project — a 3D-minimalist React+Vite site with a six-act scroll-driven scene, replacing the existing Horizons-built `neurldynamicsteam.io`, and deploy it to Hostinger.

**Architecture:** Single Vite React app. One `<Canvas>` hosts all 3D; HTML sections overlay it, synced via GSAP ScrollTrigger. All user-facing text lives in `src/content/copy.js` (edited by the Phase B prompt editor later). Scene files in `src/scene/` are hand-authored and will be locked from editor access.

**Tech Stack:** React 18, Vite 5, Tailwind CSS 3, React Three Fiber, @react-three/drei, GSAP + ScrollTrigger, framer-motion, Supabase JS (dev signup only), Vitest, Playwright.

**Working directory:** `/Users/anilkumar/neural-dynamics/` (git repo, one commit so far: the spec).

**Reference export** (READ-ONLY; copy content/assets from here, never edit in place): `/Users/anilkumar/Downloads/horizons-export-b870d9c0-abdd-4eca-b97f-879c84367732/`

**Spec:** `docs/superpowers/specs/2026-04-16-neural-dynamics-3d-redesign-and-prompt-editor-design.md`

---

## File Structure (end state)

```
neural-dynamics/
├── site/
│   ├── index.html
│   ├── package.json
│   ├── vite.config.js
│   ├── tailwind.config.js
│   ├── postcss.config.js
│   ├── .gitignore
│   ├── .nvmrc
│   ├── .env.example
│   ├── public/
│   │   ├── favicon.svg
│   │   ├── hero-fallback.webp   ← static mobile fallback (authored in Task 22)
│   │   └── models/              ← reserved for future .glb assets
│   ├── tests/
│   │   ├── vitest.setup.js
│   │   └── e2e/
│   │       └── scroll.spec.js
│   └── src/
│       ├── main.jsx
│       ├── App.jsx
│       ├── index.css
│       ├── lib/
│       │   └── supabase.js
│       ├── content/
│       │   ├── copy.js          ← source of truth for all text
│       │   └── copy.test.js
│       ├── ui/
│       │   ├── Button.jsx
│       │   ├── Input.jsx
│       │   └── Number.jsx
│       ├── scene/               ← LOCKED for editor
│       │   ├── Canvas.jsx
│       │   ├── Die.jsx
│       │   ├── Board.jsx
│       │   ├── ComputeBlocks.jsx
│       │   ├── BarColumns.jsx
│       │   ├── InterceptPath.jsx
│       │   ├── timeline.js
│       │   └── shaders/
│       │       ├── die.vert.glsl
│       │       ├── die.frag.glsl
│       │       └── grain.frag.glsl
│       ├── sections/
│       │   ├── Hero.jsx
│       │   ├── Thesis.jsx
│       │   ├── Capabilities.jsx
│       │   ├── Benchmarks.jsx
│       │   ├── DevSignup.jsx
│       │   └── Contact.jsx
│       ├── pages/
│       │   ├── Landing.jsx      ← composes sections + Canvas
│       │   ├── Blog.jsx
│       │   └── ProductDetail.jsx
│       └── hooks/
│           ├── useIsMobile.js
│           └── useScrollProgress.js
```

---

## Conventions every task follows

- Every task ends in a `git commit`.
- Every commit message uses Conventional Commits: `feat:`, `test:`, `chore:`, `fix:`, `docs:`, `style:`, `refactor:`.
- All commands run from `site/` unless a command explicitly starts with `cd ..`.
- Node version: `20.17.0` (pinned via `.nvmrc` in Task 1).
- Package manager: `npm`.
- Never edit the Horizons export folder. Always copy out, then modify.

---

## Task 1: Scaffold `site/` as a Vite + React app

**Files:**
- Create: `site/package.json`
- Create: `site/.nvmrc`
- Create: `site/.gitignore`
- Create: `site/vite.config.js`
- Create: `site/index.html`
- Create: `site/src/main.jsx`
- Create: `site/src/App.jsx`
- Create: `site/src/index.css`

- [ ] **Step 1: Create the site directory**

Run (from repo root `/Users/anilkumar/neural-dynamics`):
```bash
mkdir -p site/src site/public site/tests
```

- [ ] **Step 2: Write `.nvmrc`**

Create `site/.nvmrc`:
```
20.17.0
```

- [ ] **Step 3: Write `.gitignore`**

Create `site/.gitignore`:
```
node_modules
dist
.env
.env.local
.DS_Store
*.log
coverage
playwright-report
test-results
```

- [ ] **Step 4: Write `package.json`**

Create `site/package.json`:
```json
{
  "name": "neural-dynamics-site",
  "private": true,
  "version": "0.1.0",
  "type": "module",
  "scripts": {
    "dev": "vite",
    "build": "vite build",
    "preview": "vite preview --port 3000",
    "test": "vitest run",
    "test:watch": "vitest",
    "test:e2e": "playwright test",
    "lint": "eslint ."
  }
}
```

- [ ] **Step 5: Write `vite.config.js`**

Create `site/vite.config.js`:
```js
import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import path from 'node:path';

export default defineConfig({
  plugins: [react()],
  resolve: {
    alias: { '@': path.resolve(__dirname, 'src') },
  },
  server: { port: 3000, host: '0.0.0.0' },
  preview: { port: 3000, host: '0.0.0.0' },
  build: {
    outDir: 'dist',
    sourcemap: false,
    rollupOptions: {
      output: {
        manualChunks: {
          three: ['three', '@react-three/fiber', '@react-three/drei'],
          gsap: ['gsap'],
        },
      },
    },
  },
  assetsInclude: ['**/*.glsl'],
});
```

- [ ] **Step 6: Write `index.html`**

Create `site/index.html`:
```html
<!doctype html>
<html lang="en">
  <head>
    <meta charset="UTF-8" />
    <link rel="icon" type="image/svg+xml" href="/favicon.svg" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <meta name="theme-color" content="#000000" />
    <meta name="description" content="Neural Dynamics — GPU interception layer." />
    <title>Neural Dynamics</title>
    <link rel="preconnect" href="https://fonts.googleapis.com" />
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin />
    <link
      href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;500&display=swap"
      rel="stylesheet"
    />
  </head>
  <body class="bg-black text-neutral-50">
    <div id="root"></div>
    <script type="module" src="/src/main.jsx"></script>
  </body>
</html>
```

- [ ] **Step 7: Write `src/index.css` (base styles only — Tailwind added in Task 2)**

Create `site/src/index.css`:
```css
*,
*::before,
*::after { box-sizing: border-box; }
html, body, #root { margin: 0; padding: 0; height: 100%; }
body {
  font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
  background: #000;
  color: #fafafa;
  -webkit-font-smoothing: antialiased;
}
```

- [ ] **Step 8: Write `src/main.jsx`**

Create `site/src/main.jsx`:
```jsx
import React from 'react';
import ReactDOM from 'react-dom/client';
import App from './App.jsx';
import './index.css';

ReactDOM.createRoot(document.getElementById('root')).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>,
);
```

- [ ] **Step 9: Write placeholder `src/App.jsx`**

Create `site/src/App.jsx`:
```jsx
export default function App() {
  return (
    <main className="min-h-screen flex items-center justify-center">
      <h1 className="text-3xl">Neural Dynamics — site scaffold</h1>
    </main>
  );
}
```

- [ ] **Step 10: Install deps and verify dev server starts**

Run from `site/`:
```bash
npm install --save react@^18.3.1 react-dom@^18.3.1
npm install --save-dev vite@^5.4.0 @vitejs/plugin-react@^4.3.0
npm run dev
```
Expected: server prints `Local: http://localhost:3000`. Open it — you should see the "Neural Dynamics — site scaffold" heading. Stop the server with Ctrl-C.

- [ ] **Step 11: Commit**

Run from repo root:
```bash
git add site/
git commit -m "feat(site): scaffold vite+react app"
```

---

## Task 2: Tailwind + theme tokens

**Files:**
- Create: `site/tailwind.config.js`
- Create: `site/postcss.config.js`
- Modify: `site/src/index.css`
- Modify: `site/src/App.jsx` (temporary smoke check)

- [ ] **Step 1: Install Tailwind**

Run from `site/`:
```bash
npm install --save-dev tailwindcss@^3.4.17 postcss@^8.4.49 autoprefixer@^10.4.20
```

- [ ] **Step 2: Write `tailwind.config.js`**

Create `site/tailwind.config.js`:
```js
/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{js,jsx}'],
  theme: {
    extend: {
      colors: {
        bg: '#000000',
        ink: '#FAFAFA',
        muted: '#9A9A9A',
        amber: {
          DEFAULT: '#F5A524',
          soft: '#F5A52433',
        },
      },
      fontFamily: {
        sans: ['Inter', 'ui-sans-serif', 'system-ui'],
        mono: ['"JetBrains Mono"', 'ui-monospace', 'monospace'],
      },
      letterSpacing: {
        tightest: '-0.04em',
      },
      fontSize: {
        'display': ['clamp(3rem, 8vw, 7rem)', { lineHeight: '0.95', letterSpacing: '-0.04em' }],
        'h1': ['clamp(2rem, 4vw, 3.5rem)', { lineHeight: '1.05', letterSpacing: '-0.03em' }],
        'h2': ['clamp(1.5rem, 2.5vw, 2.25rem)', { lineHeight: '1.15', letterSpacing: '-0.02em' }],
        'lead': ['clamp(1.125rem, 1.4vw, 1.375rem)', { lineHeight: '1.5' }],
      },
    },
  },
  plugins: [],
};
```

- [ ] **Step 3: Write `postcss.config.js`**

Create `site/postcss.config.js`:
```js
export default {
  plugins: { tailwindcss: {}, autoprefixer: {} },
};
```

- [ ] **Step 4: Update `src/index.css` to use Tailwind layers**

Overwrite `site/src/index.css`:
```css
@tailwind base;
@tailwind components;
@tailwind utilities;

@layer base {
  *,
  *::before,
  *::after { box-sizing: border-box; }
  html, body, #root { margin: 0; padding: 0; height: 100%; }
  html { background: #000; color-scheme: dark; }
  body {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    background: #000;
    color: #fafafa;
    -webkit-font-smoothing: antialiased;
    text-rendering: optimizeLegibility;
  }
  ::selection { background: #F5A524; color: #000; }
}

@layer utilities {
  .font-num {
    font-family: 'JetBrains Mono', ui-monospace, monospace;
    font-variant-numeric: tabular-nums;
  }
}
```

- [ ] **Step 5: Smoke-check by updating `App.jsx`**

Overwrite `site/src/App.jsx`:
```jsx
export default function App() {
  return (
    <main className="min-h-screen flex flex-col items-center justify-center gap-6">
      <h1 className="text-display font-sans tracking-tightest">
        Neural <span className="text-amber">Dynamics</span>
      </h1>
      <p className="text-lead text-muted font-num">theme check · amber · mono numerals</p>
    </main>
  );
}
```

Run `npm run dev`, open http://localhost:3000, confirm: giant Inter headline, amber color on "Dynamics", muted mono subline. Stop server.

- [ ] **Step 6: Commit**

```bash
git add site/
git commit -m "feat(site): add tailwind theme, fonts, typography scale"
```

---

## Task 3: Routing skeleton + page shells

**Files:**
- Install: `react-router-dom`
- Create: `site/src/pages/Landing.jsx`
- Create: `site/src/pages/Blog.jsx`
- Create: `site/src/pages/ProductDetail.jsx`
- Modify: `site/src/App.jsx`

- [ ] **Step 1: Install router**

```bash
npm install --save react-router-dom@^6.26.0
```

- [ ] **Step 2: Create `src/pages/Landing.jsx`**

Create `site/src/pages/Landing.jsx`:
```jsx
export default function Landing() {
  return (
    <main className="relative">
      <h1 className="text-display tracking-tightest p-10">Landing · placeholder</h1>
    </main>
  );
}
```

- [ ] **Step 3: Create `src/pages/Blog.jsx`**

Create `site/src/pages/Blog.jsx`:
```jsx
export default function Blog() {
  return (
    <main className="max-w-3xl mx-auto px-6 py-24">
      <h1 className="text-h1">Blog</h1>
      <p className="text-muted mt-4">Posts land here.</p>
    </main>
  );
}
```

- [ ] **Step 4: Create `src/pages/ProductDetail.jsx`**

Create `site/src/pages/ProductDetail.jsx`:
```jsx
import { useParams } from 'react-router-dom';

export default function ProductDetail() {
  const { productId } = useParams();
  return (
    <main className="max-w-3xl mx-auto px-6 py-24">
      <h1 className="text-h1">Product: {productId}</h1>
    </main>
  );
}
```

- [ ] **Step 5: Overwrite `src/App.jsx` with router**

Overwrite `site/src/App.jsx`:
```jsx
import { BrowserRouter, Routes, Route } from 'react-router-dom';
import Landing from './pages/Landing.jsx';
import Blog from './pages/Blog.jsx';
import ProductDetail from './pages/ProductDetail.jsx';

export default function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<Landing />} />
        <Route path="/blog" element={<Blog />} />
        <Route path="/product/:productId" element={<ProductDetail />} />
      </Routes>
    </BrowserRouter>
  );
}
```

- [ ] **Step 6: Smoke-check all three routes**

Run `npm run dev`. Visit `/`, `/blog`, `/product/x1`. Each renders. Stop server.

- [ ] **Step 7: Commit**

```bash
git add site/
git commit -m "feat(site): add react-router with landing/blog/product routes"
```

---

## Task 4: `copy.js` content model + shape validation test (TDD)

This is the first TDD task. We define the shape of the copy data model and write a validator that future prompt-editor edits must not break.

**Files:**
- Create: `site/src/content/copy.js`
- Create: `site/src/content/copy.test.js`
- Install: `vitest`

- [ ] **Step 1: Install Vitest**

```bash
npm install --save-dev vitest@^2.0.0 jsdom@^25.0.0 @testing-library/react@^16.0.0 @testing-library/jest-dom@^6.4.0
```

- [ ] **Step 2: Add vitest config to `vite.config.js`**

Modify `site/vite.config.js` — add a `test` block below `assetsInclude`:
```js
  assetsInclude: ['**/*.glsl'],
  test: {
    environment: 'jsdom',
    globals: true,
    setupFiles: ['./tests/vitest.setup.js'],
    include: ['src/**/*.test.{js,jsx}', 'tests/**/*.test.{js,jsx}'],
    exclude: ['tests/e2e/**'],
  },
```

- [ ] **Step 3: Create `tests/vitest.setup.js`**

Create `site/tests/vitest.setup.js`:
```js
import '@testing-library/jest-dom/vitest';
```

- [ ] **Step 4: Write the failing test `src/content/copy.test.js`**

Create `site/src/content/copy.test.js`:
```js
import { describe, it, expect } from 'vitest';
import { copy } from './copy.js';

describe('copy shape', () => {
  it('has a hero with headline and subline strings', () => {
    expect(typeof copy.hero.headline).toBe('string');
    expect(copy.hero.headline.length).toBeGreaterThan(0);
    expect(typeof copy.hero.subline).toBe('string');
  });

  it('has a thesis with a body string', () => {
    expect(typeof copy.thesis.body).toBe('string');
    expect(copy.thesis.body.length).toBeGreaterThan(0);
  });

  it('has capabilities with heading and at least 3 items', () => {
    expect(typeof copy.capabilities.heading).toBe('string');
    expect(Array.isArray(copy.capabilities.items)).toBe(true);
    expect(copy.capabilities.items.length).toBeGreaterThanOrEqual(3);
    for (const item of copy.capabilities.items) {
      expect(typeof item.title).toBe('string');
      expect(typeof item.body).toBe('string');
    }
  });

  it('has benchmarks with heading and at least 3 metrics', () => {
    expect(typeof copy.benchmarks.heading).toBe('string');
    expect(Array.isArray(copy.benchmarks.metrics)).toBe(true);
    expect(copy.benchmarks.metrics.length).toBeGreaterThanOrEqual(3);
    for (const m of copy.benchmarks.metrics) {
      expect(typeof m.label).toBe('string');
      expect(typeof m.value).toBe('number');
      expect(typeof m.unit).toBe('string');
    }
  });

  it('has devSignup with heading and cta', () => {
    expect(typeof copy.devSignup.heading).toBe('string');
    expect(typeof copy.devSignup.cta).toBe('string');
  });

  it('has contact with email and socials array', () => {
    expect(typeof copy.contact.email).toBe('string');
    expect(copy.contact.email).toMatch(/@/);
    expect(Array.isArray(copy.contact.socials)).toBe(true);
  });
});
```

- [ ] **Step 5: Run test to verify it fails**

```bash
npm test
```
Expected: failure — `copy.js` does not exist.

- [ ] **Step 6: Write `src/content/copy.js` with placeholder content**

Create `site/src/content/copy.js` (real content migrated in Task 5; this is only the shape):
```js
// All user-facing text. This file is the primary surface the prompt editor edits.
// Shape is validated by copy.test.js — keep that test passing.
export const copy = {
  hero: {
    headline: 'Intercept the GPU.',
    subline: 'A thin layer between CUDA and silicon.',
  },
  thesis: {
    body:
      'Neural Dynamics builds CIPHER — a dispatch layer that sits between applications and the GPU driver, ' +
      'routing compute through interception paths that existing stacks cannot reach.',
  },
  capabilities: {
    heading: 'What CIPHER does',
    items: [
      { title: 'Intercept', body: 'Transparent to the application. No recompile. No driver changes.' },
      { title: 'Route', body: 'Dispatch kernels through custom execution paths under the driver surface.' },
      { title: 'Measure', body: 'Per-kernel telemetry at the dispatch boundary — nothing else sees it.' },
      { title: 'Compose', body: 'Chain transforms across the dispatch graph without touching user code.' },
    ],
  },
  benchmarks: {
    heading: 'Measured, not claimed',
    metrics: [
      { label: 'Intercept overhead', value: 1.8, unit: '%' },
      { label: 'Dispatch latency', value: 240, unit: 'ns' },
      { label: 'Kernels covered', value: 7, unit: '/7' },
    ],
  },
  devSignup: {
    heading: 'Early access for GPU developers',
    cta: 'Request access',
  },
  contact: {
    email: 'anil@neuraldynamics.dev',
    socials: [
      { label: 'GitHub', href: 'https://github.com/' },
      { label: 'X', href: 'https://x.com/' },
    ],
  },
};
```

- [ ] **Step 7: Run tests to verify they pass**

```bash
npm test
```
Expected: all 6 copy-shape tests pass.

- [ ] **Step 8: Commit**

```bash
git add site/
git commit -m "feat(site): add copy content model with shape validation"
```

---

## Task 5: Migrate real copy from the Horizons export

Open the old components, extract real headlines/paragraphs/numbers, and rewrite them into `copy.js`. No shape changes — we only update the string values.

**Files:**
- Modify: `site/src/content/copy.js`

**Read-only references** (extract prose from these, rewrite to be tighter):
- `/Users/anilkumar/Downloads/horizons-export-b870d9c0-abdd-4eca-b97f-879c84367732/src/components/Hero.jsx`
- `…/components/AboutSection.jsx`
- `…/components/MissionStatement.jsx`
- `…/components/CompanyThesisSection.jsx`
- `…/components/CapabilitiesGrid.jsx`
- `…/components/BenchmarksSection.jsx`
- `…/components/Benchmarks.jsx`
- `…/components/DevSignup.jsx`
- `…/components/ContactSection.jsx`
- `…/components/Footer.jsx`

- [ ] **Step 1: Read each reference component**

Use the Read tool to read each file in the bulleted list above. Note every user-facing string, benchmark number, email, and social link.

- [ ] **Step 2: Rewrite `copy.js`**

Rules for rewrites:
- **Hero headline:** ≤ 10 words. Must state the CIPHER thesis (GPU interception).
- **Hero subline:** ≤ 18 words. Technical but non-jargon.
- **Thesis body:** one paragraph, 2–3 sentences, ≤ 60 words.
- **Capabilities:** pick the 4 strongest from CapabilitiesGrid. Title ≤ 3 words, body ≤ 20 words.
- **Benchmarks:** pull the actual numbers from Benchmarks.jsx / BenchmarksSection.jsx if present; otherwise use the placeholders from Task 4 and mark them as such in a code comment. Labels ≤ 5 words.
- **DevSignup heading:** ≤ 8 words. CTA ≤ 3 words.
- **Contact:** real email from ContactSection.jsx if present (otherwise `anil.0666369@gmail.com` per user email in the env). Socials: GitHub + any linked in the existing Footer.jsx.

Overwrite `site/src/content/copy.js` with the migrated values, preserving the exact shape from Task 4. If any benchmark number in the old site was a placeholder, leave it as a placeholder here and add a `// PLACEHOLDER — confirm with real CIPHER data` comment on that metric's line.

- [ ] **Step 3: Run copy tests**

```bash
npm test
```
Expected: all 6 tests still pass.

- [ ] **Step 4: Commit**

```bash
git add site/src/content/copy.js
git commit -m "feat(site): migrate copy from horizons export into copy.js"
```

---

## Task 6: Supabase client + shared UI primitives

**Files:**
- Install: `@supabase/supabase-js`
- Create: `site/.env.example`
- Create: `site/src/lib/supabase.js`
- Create: `site/src/ui/Button.jsx`
- Create: `site/src/ui/Input.jsx`
- Create: `site/src/ui/Number.jsx`
- Create: `site/src/ui/Number.test.jsx`

- [ ] **Step 1: Install Supabase JS**

```bash
npm install --save @supabase/supabase-js@^2.30.0
```

- [ ] **Step 2: Write `.env.example`**

Create `site/.env.example`:
```
VITE_SUPABASE_URL=
VITE_SUPABASE_ANON_KEY=
```

(Copy the real values from the Horizons export's `.env` if present; otherwise these can stay empty until DevSignup is wired. DevSignup must gracefully no-op when keys are missing — see Task 20.)

- [ ] **Step 3: Write `src/lib/supabase.js`**

Create `site/src/lib/supabase.js`:
```js
import { createClient } from '@supabase/supabase-js';

const url = import.meta.env.VITE_SUPABASE_URL;
const anonKey = import.meta.env.VITE_SUPABASE_ANON_KEY;

export const supabase = url && anonKey ? createClient(url, anonKey) : null;
export const supabaseReady = Boolean(supabase);
```

- [ ] **Step 4: Write `src/ui/Button.jsx`**

Create `site/src/ui/Button.jsx`:
```jsx
export default function Button({
  children,
  onClick,
  type = 'button',
  variant = 'primary',
  disabled = false,
  className = '',
  ...rest
}) {
  const base =
    'inline-flex items-center justify-center px-5 py-2.5 text-sm font-medium rounded-full ' +
    'transition-[transform,background,color,border] duration-200 outline-none ' +
    'focus-visible:ring-2 focus-visible:ring-amber focus-visible:ring-offset-2 focus-visible:ring-offset-bg ' +
    'disabled:opacity-40 disabled:cursor-not-allowed';
  const variants = {
    primary: 'bg-amber text-black hover:-translate-y-0.5 active:translate-y-0',
    ghost: 'bg-transparent text-ink border border-neutral-800 hover:border-amber hover:text-amber',
  };
  return (
    <button
      type={type}
      onClick={onClick}
      disabled={disabled}
      className={`${base} ${variants[variant]} ${className}`}
      {...rest}
    >
      {children}
    </button>
  );
}
```

- [ ] **Step 5: Write `src/ui/Input.jsx`**

Create `site/src/ui/Input.jsx`:
```jsx
export default function Input({ className = '', ...rest }) {
  return (
    <input
      {...rest}
      className={
        'w-full bg-transparent border-b border-neutral-800 px-0 py-3 text-ink ' +
        'placeholder:text-muted focus:outline-none focus:border-amber transition-colors ' +
        className
      }
    />
  );
}
```

- [ ] **Step 6: Write failing test for `Number` component**

Create `site/src/ui/Number.test.jsx`:
```jsx
import { describe, it, expect } from 'vitest';
import { render, screen } from '@testing-library/react';
import Number from './Number.jsx';

describe('Number', () => {
  it('renders integer values as tabular mono numerals', () => {
    render(<Number value={240} unit="ns" label="latency" />);
    expect(screen.getByText('240')).toBeInTheDocument();
    expect(screen.getByText('ns')).toBeInTheDocument();
    expect(screen.getByText('latency')).toBeInTheDocument();
  });

  it('renders one decimal place when value is fractional', () => {
    render(<Number value={1.8} unit="%" label="overhead" />);
    expect(screen.getByText('1.8')).toBeInTheDocument();
  });
});
```

- [ ] **Step 7: Run the test to verify it fails**

```bash
npm test src/ui/Number.test.jsx
```
Expected: failure — `Number.jsx` does not exist.

- [ ] **Step 8: Write `src/ui/Number.jsx`**

Create `site/src/ui/Number.jsx`:
```jsx
function formatValue(value) {
  if (Number.isInteger(value)) return String(value);
  return value.toFixed(1);
}

export default function Number({ value, unit = '', label = '', className = '' }) {
  return (
    <div className={`flex flex-col ${className}`}>
      <div className="flex items-baseline gap-1 font-num text-ink">
        <span className="text-[clamp(2.5rem,5vw,4.5rem)] leading-none">{formatValue(value)}</span>
        {unit && <span className="text-muted text-lg">{unit}</span>}
      </div>
      {label && <div className="mt-2 text-muted text-sm uppercase tracking-wider">{label}</div>}
    </div>
  );
}
```

- [ ] **Step 9: Run the test to verify it passes**

```bash
npm test src/ui/Number.test.jsx
```
Expected: both tests pass.

- [ ] **Step 10: Commit**

```bash
git add site/
git commit -m "feat(site): add supabase client and shared ui primitives"
```

---

## Task 7: `useIsMobile` + `useScrollProgress` hooks

Mobile detection disables WebGL (per spec §5.6). Scroll progress drives the 3D timeline.

**Files:**
- Create: `site/src/hooks/useIsMobile.js`
- Create: `site/src/hooks/useIsMobile.test.jsx`
- Create: `site/src/hooks/useScrollProgress.js`

- [ ] **Step 1: Write failing test for `useIsMobile`**

Create `site/src/hooks/useIsMobile.test.jsx`:
```jsx
import { describe, it, expect, beforeEach } from 'vitest';
import { renderHook } from '@testing-library/react';
import useIsMobile from './useIsMobile.js';

function mockMatchMedia(matches) {
  window.matchMedia = (q) => ({
    matches,
    media: q,
    onchange: null,
    addEventListener: () => {},
    removeEventListener: () => {},
    addListener: () => {},
    removeListener: () => {},
    dispatchEvent: () => false,
  });
}

describe('useIsMobile', () => {
  beforeEach(() => { mockMatchMedia(false); });

  it('returns false on wide viewports', () => {
    mockMatchMedia(false);
    const { result } = renderHook(() => useIsMobile());
    expect(result.current).toBe(false);
  });

  it('returns true on narrow viewports', () => {
    mockMatchMedia(true);
    const { result } = renderHook(() => useIsMobile());
    expect(result.current).toBe(true);
  });
});
```

- [ ] **Step 2: Run to verify failure**

```bash
npm test src/hooks/useIsMobile.test.jsx
```
Expected: failure — file not found.

- [ ] **Step 3: Write `useIsMobile.js`**

Create `site/src/hooks/useIsMobile.js`:
```js
import { useEffect, useState } from 'react';

const QUERY = '(max-width: 767px)';

export default function useIsMobile() {
  const [mobile, setMobile] = useState(() =>
    typeof window === 'undefined' ? false : window.matchMedia(QUERY).matches,
  );

  useEffect(() => {
    if (typeof window === 'undefined') return;
    const mql = window.matchMedia(QUERY);
    const onChange = (e) => setMobile(e.matches);
    mql.addEventListener?.('change', onChange);
    setMobile(mql.matches);
    return () => mql.removeEventListener?.('change', onChange);
  }, []);

  return mobile;
}
```

- [ ] **Step 4: Run tests to verify pass**

```bash
npm test src/hooks/useIsMobile.test.jsx
```
Expected: both tests pass.

- [ ] **Step 5: Write `useScrollProgress.js`**

Create `site/src/hooks/useScrollProgress.js`:
```js
import { useEffect, useState } from 'react';

// Returns a number in [0,1] for whole-page scroll.
// For per-section ranges the 3D timeline uses GSAP ScrollTrigger directly.
export default function useScrollProgress() {
  const [p, setP] = useState(0);

  useEffect(() => {
    if (typeof window === 'undefined') return;
    const onScroll = () => {
      const doc = document.documentElement;
      const max = (doc.scrollHeight - window.innerHeight) || 1;
      setP(Math.min(1, Math.max(0, window.scrollY / max)));
    };
    onScroll();
    window.addEventListener('scroll', onScroll, { passive: true });
    window.addEventListener('resize', onScroll);
    return () => {
      window.removeEventListener('scroll', onScroll);
      window.removeEventListener('resize', onScroll);
    };
  }, []);

  return p;
}
```

- [ ] **Step 6: Commit**

```bash
git add site/
git commit -m "feat(site): add useIsMobile and useScrollProgress hooks"
```

---

## Task 8: R3F install + Canvas root

**Files:**
- Install: `three`, `@react-three/fiber`, `@react-three/drei`, `gsap`
- Create: `site/src/scene/Canvas.jsx`
- Create: `site/src/scene/timeline.js`

- [ ] **Step 1: Install 3D stack**

```bash
npm install --save three@^0.168.0 @react-three/fiber@^8.17.0 @react-three/drei@^9.114.0 gsap@^3.12.5 framer-motion@^11.15.0
```

- [ ] **Step 2: Write `src/scene/timeline.js` (skeleton — acts added in Tasks 11–15)**

Create `site/src/scene/timeline.js`:
```js
import { gsap } from 'gsap';
import { ScrollTrigger } from 'gsap/ScrollTrigger';

gsap.registerPlugin(ScrollTrigger);

// Section ids in order — must match the id attributes on each <section> in pages/Landing.jsx.
export const SECTION_IDS = ['hero', 'thesis', 'capabilities', 'benchmarks', 'devsignup', 'contact'];

// Returns a master GSAP timeline that scrubs with the whole page scroll.
// Individual 3D components register their own tweens into this timeline via `register(tl, refs)`.
export function createMasterTimeline(registrations) {
  const tl = gsap.timeline({
    scrollTrigger: {
      trigger: document.body,
      start: 'top top',
      end: 'bottom bottom',
      scrub: 1,
      invalidateOnRefresh: true,
    },
  });
  for (const registration of registrations) {
    if (typeof registration === 'function') registration(tl);
  }
  return tl;
}

export function refresh() { ScrollTrigger.refresh(); }
```

- [ ] **Step 3: Write `src/scene/Canvas.jsx`**

Create `site/src/scene/Canvas.jsx`:
```jsx
import { Suspense } from 'react';
import { Canvas as R3FCanvas } from '@react-three/fiber';
import { PerspectiveCamera } from '@react-three/drei';

export default function Canvas({ children }) {
  return (
    <div
      aria-hidden="true"
      className="fixed inset-0 -z-10 pointer-events-none"
      style={{ background: '#000' }}
    >
      <R3FCanvas
        dpr={[1, 1.75]}
        gl={{ antialias: true, alpha: false, powerPreference: 'high-performance' }}
        camera={{ fov: 35, near: 0.1, far: 100 }}
      >
        <PerspectiveCamera makeDefault position={[0, 0, 6]} fov={35} />
        <color attach="background" args={['#000000']} />
        <ambientLight intensity={0.4} />
        <directionalLight position={[3, 5, 2]} intensity={1.5} color="#ffffff" />
        <directionalLight position={[-4, 2, -3]} intensity={0.6} color="#F5A524" />
        <Suspense fallback={null}>{children}</Suspense>
      </R3FCanvas>
    </div>
  );
}
```

- [ ] **Step 4: Drop Canvas into Landing for a smoke check**

Overwrite `site/src/pages/Landing.jsx`:
```jsx
import Canvas from '@/scene/Canvas.jsx';

export default function Landing() {
  return (
    <>
      <Canvas />
      <main className="relative">
        <section id="hero" className="h-screen flex items-center justify-center">
          <h1 className="text-display tracking-tightest">Scene canvas mounted</h1>
        </section>
        <section id="thesis" className="h-screen" />
        <section id="capabilities" className="h-screen" />
        <section id="benchmarks" className="h-screen" />
        <section id="devsignup" className="h-screen" />
        <section id="contact" className="h-screen" />
      </main>
    </>
  );
}
```

- [ ] **Step 5: Smoke run**

```bash
npm run dev
```
Open `http://localhost:3000`. Expected: black background (R3F canvas active), heading visible on top. Scroll through — sections are empty but scroll works. Stop server.

- [ ] **Step 6: Commit**

```bash
git add site/
git commit -m "feat(site): add r3f canvas root and scroll timeline skeleton"
```

---

## Task 9: GLSL shader assets

**Files:**
- Create: `site/src/scene/shaders/die.vert.glsl`
- Create: `site/src/scene/shaders/die.frag.glsl`
- Create: `site/src/scene/shaders/grain.frag.glsl`

- [ ] **Step 1: Write `die.vert.glsl`**

Create `site/src/scene/shaders/die.vert.glsl`:
```glsl
varying vec3 vNormal;
varying vec3 vWorldPosition;
varying vec2 vUv;

void main() {
  vUv = uv;
  vNormal = normalize(normalMatrix * normal);
  vec4 worldPos = modelMatrix * vec4(position, 1.0);
  vWorldPosition = worldPos.xyz;
  gl_Position = projectionMatrix * viewMatrix * worldPos;
}
```

- [ ] **Step 2: Write `die.frag.glsl`**

Create `site/src/scene/shaders/die.frag.glsl`:
```glsl
precision highp float;

uniform float uTime;
uniform vec3  uBaseColor;
uniform vec3  uAccent;
uniform float uAccentAmount;

varying vec3 vNormal;
varying vec3 vWorldPosition;
varying vec2 vUv;

// Simple grid — evokes a die's lithography pattern.
float grid(vec2 uv, float cells) {
  vec2 g = fract(uv * cells);
  float line = min(g.x, g.y);
  line = min(line, 1.0 - max(g.x, g.y));
  return smoothstep(0.0, 0.02, line);
}

void main() {
  vec3 viewDir = normalize(cameraPosition - vWorldPosition);
  float fresnel = pow(1.0 - max(dot(normalize(vNormal), viewDir), 0.0), 3.0);

  float gridMask = grid(vUv, 24.0);
  vec3 base = mix(uBaseColor * 0.2, uBaseColor, gridMask);

  vec3 rim = uAccent * fresnel * uAccentAmount;
  vec3 color = base + rim;

  gl_FragColor = vec4(color, 1.0);
}
```

- [ ] **Step 3: Write `grain.frag.glsl`** (kept for later use as a post-process; safe to author now)

Create `site/src/scene/shaders/grain.frag.glsl`:
```glsl
precision highp float;
uniform sampler2D tDiffuse;
uniform float uTime;
varying vec2 vUv;

float rand(vec2 co) {
  return fract(sin(dot(co.xy, vec2(12.9898, 78.233))) * 43758.5453);
}

void main() {
  vec4 tex = texture2D(tDiffuse, vUv);
  float n = rand(vUv * (uTime + 1.0)) - 0.5;
  gl_FragColor = vec4(tex.rgb + n * 0.04, tex.a);
}
```

- [ ] **Step 4: Commit**

```bash
git add site/
git commit -m "feat(site): add die vertex/fragment and grain shaders"
```

---

## Task 10: `Die` mesh (Act 1 — hero)

**Files:**
- Create: `site/src/scene/Die.jsx`

- [ ] **Step 1: Write `Die.jsx`**

Create `site/src/scene/Die.jsx`:
```jsx
import { useRef, useMemo } from 'react';
import { useFrame } from '@react-three/fiber';
import * as THREE from 'three';
import vert from './shaders/die.vert.glsl?raw';
import frag from './shaders/die.frag.glsl?raw';

export default function Die({ position = [0, 0, 0], rotationSpeed = 0.05 }) {
  const meshRef = useRef(null);

  const material = useMemo(
    () =>
      new THREE.ShaderMaterial({
        vertexShader: vert,
        fragmentShader: frag,
        uniforms: {
          uTime: { value: 0 },
          uBaseColor: { value: new THREE.Color('#8a8a8a') },
          uAccent: { value: new THREE.Color('#F5A524') },
          uAccentAmount: { value: 1.1 },
        },
      }),
    [],
  );

  useFrame((state) => {
    if (!meshRef.current) return;
    meshRef.current.rotation.y += rotationSpeed * 0.016;
    material.uniforms.uTime.value = state.clock.elapsedTime;
  });

  return (
    <mesh ref={meshRef} position={position} material={material}>
      <boxGeometry args={[1.6, 0.08, 1.6, 1, 1, 1]} />
    </mesh>
  );
}
```

- [ ] **Step 2: Mount it in Canvas for visual check**

Modify `site/src/pages/Landing.jsx` — import and render the die inside the Canvas:
```jsx
import Canvas from '@/scene/Canvas.jsx';
import Die from '@/scene/Die.jsx';

export default function Landing() {
  return (
    <>
      <Canvas>
        <Die position={[0, 0, 0]} />
      </Canvas>
      <main className="relative">
        <section id="hero" className="h-screen flex items-center justify-center">
          <h1 className="text-display tracking-tightest">Hero</h1>
        </section>
        <section id="thesis" className="h-screen" />
        <section id="capabilities" className="h-screen" />
        <section id="benchmarks" className="h-screen" />
        <section id="devsignup" className="h-screen" />
        <section id="contact" className="h-screen" />
      </main>
    </>
  );
}
```

- [ ] **Step 3: Visual check**

```bash
npm run dev
```
Expected: a flat silicon-die rectangle rotating slowly, amber rim-light visible at grazing angles, faint lithography grid pattern on the top surface. Stop server.

- [ ] **Step 4: Commit**

```bash
git add site/
git commit -m "feat(site): add silicon die mesh with shader material"
```

---

## Task 11: `Board` formation (Act 2 — thesis)

The board is a slightly larger PCB that the die sits on. Camera animation between Act 1 and Act 2 pulls back to reveal it.

**Files:**
- Create: `site/src/scene/Board.jsx`
- Modify: `site/src/scene/timeline.js` (register camera pullback — added in Task 16 when all acts tie together)

- [ ] **Step 1: Write `Board.jsx`**

Create `site/src/scene/Board.jsx`:
```jsx
import { useRef } from 'react';
import { useFrame } from '@react-three/fiber';
import * as THREE from 'three';

export default function Board({ position = [0, -0.2, 0], opacity = 1 }) {
  const groupRef = useRef(null);

  useFrame(() => {
    if (!groupRef.current) return;
    groupRef.current.traverse((obj) => {
      if (obj.isMesh && obj.material) {
        obj.material.opacity = opacity;
        obj.material.transparent = opacity < 1;
      }
    });
  });

  return (
    <group ref={groupRef} position={position}>
      {/* Main PCB */}
      <mesh position={[0, 0, 0]}>
        <boxGeometry args={[5, 0.04, 3.5]} />
        <meshStandardMaterial color="#0b0b0d" metalness={0.2} roughness={0.9} />
      </mesh>
      {/* Traces — thin amber lines on the board surface */}
      {Array.from({ length: 6 }).map((_, i) => (
        <mesh key={i} position={[-1.8 + i * 0.7, 0.022, 0]}>
          <boxGeometry args={[0.02, 0.001, 3]} />
          <meshBasicMaterial color="#F5A524" transparent opacity={0.55} />
        </mesh>
      ))}
      {/* Solder pads */}
      {Array.from({ length: 8 }).map((_, i) => (
        <mesh key={`pad-${i}`} position={[-2 + (i % 4) * 1.3, 0.022, i < 4 ? 1.3 : -1.3]}>
          <cylinderGeometry args={[0.06, 0.06, 0.002, 16]} />
          <meshStandardMaterial color="#2a2a2f" metalness={0.9} roughness={0.3} />
        </mesh>
      ))}
    </group>
  );
}
```

- [ ] **Step 2: Visual check with die on board**

Modify `Landing.jsx`:
```jsx
<Canvas>
  <Die position={[0, 0.06, 0]} />
  <Board position={[0, -0.05, 0]} />
</Canvas>
```
Run `npm run dev`. Expected: die visible on top of a dark PCB with 6 amber traces. Stop.

- [ ] **Step 3: Commit**

```bash
git add site/
git commit -m "feat(site): add gpu board formation for act 2"
```

---

## Task 12: `ComputeBlocks` (Act 3 — capabilities)

Four floating cubes representing capabilities. Each one surfaces/pops forward as scroll passes its anchor.

**Files:**
- Create: `site/src/scene/ComputeBlocks.jsx`

- [ ] **Step 1: Write `ComputeBlocks.jsx`**

Create `site/src/scene/ComputeBlocks.jsx`:
```jsx
import { useMemo, useRef } from 'react';
import { useFrame } from '@react-three/fiber';
import * as THREE from 'three';

const BLOCKS = [
  { x: -2.0, y: 0.4, z: 0.6, delay: 0.0 },
  { x: -0.7, y: 0.8, z: -0.3, delay: 0.2 },
  { x: 0.9, y: 0.5, z: 0.3, delay: 0.4 },
  { x: 2.1, y: 0.9, z: -0.6, delay: 0.6 },
];

export default function ComputeBlocks({ progress = 0 }) {
  const refs = useRef(BLOCKS.map(() => ({ mesh: null })));

  useFrame((state) => {
    for (let i = 0; i < BLOCKS.length; i++) {
      const ref = refs.current[i].mesh;
      if (!ref) continue;
      const local = Math.min(1, Math.max(0, (progress - BLOCKS[i].delay) * 2));
      ref.scale.setScalar(THREE.MathUtils.lerp(0.01, 1, local));
      ref.rotation.y = state.clock.elapsedTime * 0.2 + i;
      ref.rotation.x = state.clock.elapsedTime * 0.1 + i * 0.5;
    }
  });

  return (
    <group>
      {BLOCKS.map((b, i) => (
        <mesh
          key={i}
          ref={(el) => (refs.current[i].mesh = el)}
          position={[b.x, b.y, b.z]}
        >
          <boxGeometry args={[0.35, 0.35, 0.35]} />
          <meshStandardMaterial
            color="#ffffff"
            emissive="#F5A524"
            emissiveIntensity={0.15}
            metalness={0.4}
            roughness={0.5}
          />
        </mesh>
      ))}
    </group>
  );
}
```

- [ ] **Step 2: Commit**

```bash
git add site/
git commit -m "feat(site): add compute blocks group for act 3"
```

---

## Task 13: `BarColumns` (Act 4 — benchmarks)

Three 3D bar columns animated to match benchmark metrics.

**Files:**
- Create: `site/src/scene/BarColumns.jsx`

- [ ] **Step 1: Write `BarColumns.jsx`**

Create `site/src/scene/BarColumns.jsx`:
```jsx
import { useRef } from 'react';
import { useFrame } from '@react-three/fiber';
import * as THREE from 'three';

// Heights are normalized 0..1 — the actual benchmark numbers are shown as HTML text alongside.
const BARS = [
  { x: -1.4, targetHeight: 0.25 }, // intercept overhead (low is good — short bar)
  { x: 0.0,  targetHeight: 0.6  }, // dispatch latency
  { x: 1.4,  targetHeight: 1.0  }, // kernels covered
];

export default function BarColumns({ progress = 0 }) {
  const refs = useRef(BARS.map(() => ({ mesh: null })));

  useFrame(() => {
    for (let i = 0; i < BARS.length; i++) {
      const m = refs.current[i].mesh;
      if (!m) continue;
      const local = Math.min(1, Math.max(0, progress * 1.5 - i * 0.1));
      const h = THREE.MathUtils.lerp(0.0, BARS[i].targetHeight * 2.2, local);
      m.scale.y = Math.max(0.001, h);
      m.position.y = h / 2;
    }
  });

  return (
    <group>
      {BARS.map((b, i) => (
        <mesh
          key={i}
          ref={(el) => (refs.current[i].mesh = el)}
          position={[b.x, 0, 0]}
        >
          <boxGeometry args={[0.55, 1, 0.55]} />
          <meshStandardMaterial
            color="#F5A524"
            emissive="#F5A524"
            emissiveIntensity={0.4}
            metalness={0.1}
            roughness={0.4}
          />
        </mesh>
      ))}
    </group>
  );
}
```

- [ ] **Step 2: Commit**

```bash
git add site/
git commit -m "feat(site): add benchmark bar columns for act 4"
```

---

## Task 14: `InterceptPath` (Act 5 — dev signup)

A single amber line cutting across the screen — visual anchor for dev signup.

**Files:**
- Create: `site/src/scene/InterceptPath.jsx`

- [ ] **Step 1: Write `InterceptPath.jsx`**

Create `site/src/scene/InterceptPath.jsx`:
```jsx
import { useMemo, useRef } from 'react';
import { useFrame } from '@react-three/fiber';
import * as THREE from 'three';

export default function InterceptPath({ progress = 0 }) {
  const lineRef = useRef(null);
  const material = useMemo(
    () => new THREE.LineBasicMaterial({ color: '#F5A524', transparent: true, opacity: 0 }),
    [],
  );

  const geometry = useMemo(() => {
    const g = new THREE.BufferGeometry();
    const pts = [];
    for (let i = 0; i <= 64; i++) {
      const t = i / 64;
      const x = THREE.MathUtils.lerp(-5, 5, t);
      const y = Math.sin(t * Math.PI * 2) * 0.15;
      const z = 0;
      pts.push(x, y, z);
    }
    g.setAttribute('position', new THREE.Float32BufferAttribute(pts, 3));
    g.setDrawRange(0, 0);
    return g;
  }, []);

  useFrame(() => {
    const drawCount = Math.floor(65 * progress);
    geometry.setDrawRange(0, drawCount);
    material.opacity = Math.min(1, progress * 2);
  });

  return <line ref={lineRef} geometry={geometry} material={material} />;
}
```

- [ ] **Step 2: Commit**

```bash
git add site/
git commit -m "feat(site): add intercept-path line for act 5"
```

---

## Task 15: Master timeline wiring all acts

This task composes the Canvas to consume scroll progress and render the right 3D elements with the right camera position per section.

**Files:**
- Modify: `site/src/scene/Canvas.jsx`
- Modify: `site/src/scene/timeline.js`

- [ ] **Step 1: Extend `timeline.js` with section-progress helpers**

Overwrite `site/src/scene/timeline.js`:
```js
import { gsap } from 'gsap';
import { ScrollTrigger } from 'gsap/ScrollTrigger';

gsap.registerPlugin(ScrollTrigger);

export const SECTION_IDS = ['hero', 'thesis', 'capabilities', 'benchmarks', 'devsignup', 'contact'];

// Build a ScrollTrigger per section that writes its 0..1 progress into a shared object.
export function attachSectionProgress(store) {
  const triggers = SECTION_IDS.map((id) =>
    ScrollTrigger.create({
      trigger: `#${id}`,
      start: 'top bottom',
      end: 'bottom top',
      onUpdate: (self) => { store[id] = self.progress; },
    }),
  );
  return () => triggers.forEach((t) => t.kill());
}

export function refresh() { ScrollTrigger.refresh(); }
```

- [ ] **Step 2: Rewrite `Canvas.jsx` to consume scroll state**

Overwrite `site/src/scene/Canvas.jsx`:
```jsx
import { Suspense, useEffect, useRef, useState } from 'react';
import { Canvas as R3FCanvas, useFrame, useThree } from '@react-three/fiber';
import { attachSectionProgress, SECTION_IDS } from './timeline.js';
import Die from './Die.jsx';
import Board from './Board.jsx';
import ComputeBlocks from './ComputeBlocks.jsx';
import BarColumns from './BarColumns.jsx';
import InterceptPath from './InterceptPath.jsx';

function CameraRig({ progress }) {
  const { camera } = useThree();
  useFrame(() => {
    // Keyframed camera path across 6 sections (0..5 progress).
    const p = progress.current;
    // Act 1: close on die. Act 2: pull back. Act 3: lift up. Act 4: front-on. Act 5-6: recede.
    const zTarget = 2.4 + p * 1.8;
    const yTarget = 0.1 + Math.min(p, 3) * 0.3;
    camera.position.x += (0 - camera.position.x) * 0.08;
    camera.position.y += (yTarget - camera.position.y) * 0.08;
    camera.position.z += (zTarget - camera.position.z) * 0.08;
    camera.lookAt(0, 0, 0);
  });
  return null;
}

function Stage({ store }) {
  const progress = useRef(0);
  useFrame(() => {
    // Whole-page progress ~ sum of section progresses / N
    let total = 0;
    for (const id of SECTION_IDS) total += store[id] || 0;
    progress.current = Math.min(SECTION_IDS.length, total);
  });

  return (
    <>
      <CameraRig progress={progress} />
      <Die />
      <Board opacity={Math.min(1, (store.thesis || 0) + (store.capabilities || 0))} />
      <ComputeBlocks progress={store.capabilities || 0} />
      <group position={[0, -0.4, 0]} visible={(store.benchmarks || 0) > 0.02}>
        <BarColumns progress={store.benchmarks || 0} />
      </group>
      <group visible={(store.devsignup || 0) > 0.02}>
        <InterceptPath progress={store.devsignup || 0} />
      </group>
    </>
  );
}

export default function Canvas() {
  const [store] = useState(() => ({}));

  useEffect(() => {
    const detach = attachSectionProgress(store);
    return () => detach();
  }, [store]);

  return (
    <div aria-hidden="true" className="fixed inset-0 -z-10 pointer-events-none" style={{ background: '#000' }}>
      <R3FCanvas
        dpr={[1, 1.75]}
        gl={{ antialias: true, alpha: false, powerPreference: 'high-performance' }}
        camera={{ fov: 35, near: 0.1, far: 100, position: [0, 0, 2.4] }}
      >
        <color attach="background" args={['#000000']} />
        <ambientLight intensity={0.4} />
        <directionalLight position={[3, 5, 2]} intensity={1.5} color="#ffffff" />
        <directionalLight position={[-4, 2, -3]} intensity={0.6} color="#F5A524" />
        <Suspense fallback={null}>
          <Stage store={store} />
        </Suspense>
      </R3FCanvas>
    </div>
  );
}
```

- [ ] **Step 3: Smoke scroll**

Run `npm run dev`, scroll all the way down. Expected: die is visible in hero; board fades in at thesis; blocks pop in at capabilities; bars rise at benchmarks; amber line draws at devsignup. No crashes. Stop.

- [ ] **Step 4: Commit**

```bash
git add site/
git commit -m "feat(site): wire master timeline and camera rig for all acts"
```

---

## Task 16: `Hero` section (HTML overlay)

**Files:**
- Create: `site/src/sections/Hero.jsx`
- Modify: `site/src/pages/Landing.jsx`

- [ ] **Step 1: Write `Hero.jsx`**

Create `site/src/sections/Hero.jsx`:
```jsx
import { copy } from '@/content/copy.js';

export default function Hero() {
  return (
    <section
      id="hero"
      className="h-screen flex flex-col justify-center px-8 md:px-16 max-w-6xl mx-auto"
    >
      <h1 className="text-display tracking-tightest font-sans">
        {copy.hero.headline}
      </h1>
      <p className="mt-6 text-lead text-muted font-num max-w-xl">
        {copy.hero.subline}
      </p>
    </section>
  );
}
```

- [ ] **Step 2: Mount in Landing**

Overwrite `site/src/pages/Landing.jsx`:
```jsx
import Canvas from '@/scene/Canvas.jsx';
import Hero from '@/sections/Hero.jsx';

export default function Landing() {
  return (
    <>
      <Canvas />
      <main className="relative">
        <Hero />
        <section id="thesis" className="h-screen" />
        <section id="capabilities" className="h-screen" />
        <section id="benchmarks" className="h-screen" />
        <section id="devsignup" className="h-screen" />
        <section id="contact" className="h-screen" />
      </main>
    </>
  );
}
```

- [ ] **Step 3: Commit**

```bash
git add site/
git commit -m "feat(site): add hero section with copy-driven headline"
```

---

## Task 17: `Thesis` section

**Files:**
- Create: `site/src/sections/Thesis.jsx`
- Modify: `site/src/pages/Landing.jsx`

- [ ] **Step 1: Write `Thesis.jsx`**

Create `site/src/sections/Thesis.jsx`:
```jsx
import { copy } from '@/content/copy.js';

export default function Thesis() {
  return (
    <section
      id="thesis"
      className="min-h-screen flex items-center px-8 md:px-16 max-w-6xl mx-auto"
    >
      <div className="max-w-2xl">
        <div className="text-sm uppercase tracking-widest text-amber font-num mb-6">
          Thesis
        </div>
        <p className="text-h2 font-sans text-ink leading-snug">
          {copy.thesis.body}
        </p>
      </div>
    </section>
  );
}
```

- [ ] **Step 2: Mount it**

Modify `site/src/pages/Landing.jsx` to import and render `<Thesis />` in place of the empty `<section id="thesis" />`.

- [ ] **Step 3: Commit**

```bash
git add site/
git commit -m "feat(site): add thesis section"
```

---

## Task 18: `Capabilities` section

**Files:**
- Create: `site/src/sections/Capabilities.jsx`
- Modify: `site/src/pages/Landing.jsx`

- [ ] **Step 1: Write `Capabilities.jsx`**

Create `site/src/sections/Capabilities.jsx`:
```jsx
import { copy } from '@/content/copy.js';

export default function Capabilities() {
  return (
    <section
      id="capabilities"
      className="min-h-screen flex items-center px-8 md:px-16 max-w-6xl mx-auto"
    >
      <div className="w-full">
        <div className="text-sm uppercase tracking-widest text-amber font-num mb-6">
          Capabilities
        </div>
        <h2 className="text-h1 tracking-tightest mb-12 max-w-2xl">
          {copy.capabilities.heading}
        </h2>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-x-16 gap-y-10 max-w-4xl">
          {copy.capabilities.items.map((item, i) => (
            <div key={i} className="flex gap-6">
              <div className="font-num text-amber text-sm pt-1">
                0{i + 1}
              </div>
              <div>
                <h3 className="text-h2 font-sans mb-2">{item.title}</h3>
                <p className="text-muted text-lead">{item.body}</p>
              </div>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}
```

- [ ] **Step 2: Mount in Landing**

Modify `Landing.jsx` — import and render `<Capabilities />`.

- [ ] **Step 3: Commit**

```bash
git add site/
git commit -m "feat(site): add capabilities section"
```

---

## Task 19: `Benchmarks` section (with count-up)

**Files:**
- Create: `site/src/sections/Benchmarks.jsx`
- Modify: `site/src/pages/Landing.jsx`

- [ ] **Step 1: Write `Benchmarks.jsx`**

Create `site/src/sections/Benchmarks.jsx`:
```jsx
import { useEffect, useRef, useState } from 'react';
import { copy } from '@/content/copy.js';
import Number from '@/ui/Number.jsx';

function useInView() {
  const ref = useRef(null);
  const [inView, setInView] = useState(false);
  useEffect(() => {
    if (!ref.current) return;
    const obs = new IntersectionObserver(
      ([e]) => { if (e.isIntersecting) setInView(true); },
      { threshold: 0.35 },
    );
    obs.observe(ref.current);
    return () => obs.disconnect();
  }, []);
  return [ref, inView];
}

function useCountUp(target, run) {
  const [n, setN] = useState(0);
  useEffect(() => {
    if (!run) return;
    let raf;
    const start = performance.now();
    const duration = 900;
    const tick = (t) => {
      const p = Math.min(1, (t - start) / duration);
      setN(target * (1 - Math.pow(1 - p, 3)));
      if (p < 1) raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [target, run]);
  return n;
}

function Metric({ label, value, unit, run }) {
  const n = useCountUp(value, run);
  const display = Number.isInteger(value) ? Math.round(n) : Math.round(n * 10) / 10;
  return <Number value={display} unit={unit} label={label} />;
}

export default function Benchmarks() {
  const [ref, inView] = useInView();
  return (
    <section
      id="benchmarks"
      ref={ref}
      className="min-h-screen flex items-center px-8 md:px-16 max-w-6xl mx-auto"
    >
      <div className="w-full">
        <div className="text-sm uppercase tracking-widest text-amber font-num mb-6">
          Benchmarks
        </div>
        <h2 className="text-h1 tracking-tightest mb-12 max-w-2xl">
          {copy.benchmarks.heading}
        </h2>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-12">
          {copy.benchmarks.metrics.map((m, i) => (
            <Metric key={i} label={m.label} value={m.value} unit={m.unit} run={inView} />
          ))}
        </div>
      </div>
    </section>
  );
}
```

- [ ] **Step 2: Mount in Landing**

Modify `Landing.jsx` — import and render `<Benchmarks />`.

- [ ] **Step 3: Commit**

```bash
git add site/
git commit -m "feat(site): add benchmarks section with count-up numerals"
```

---

## Task 20: `DevSignup` section (Supabase-backed form)

**Files:**
- Create: `site/src/sections/DevSignup.jsx`
- Modify: `site/src/pages/Landing.jsx`

- [ ] **Step 1: Write `DevSignup.jsx`**

Create `site/src/sections/DevSignup.jsx`:
```jsx
import { useState } from 'react';
import { copy } from '@/content/copy.js';
import { supabase, supabaseReady } from '@/lib/supabase.js';
import Button from '@/ui/Button.jsx';
import Input from '@/ui/Input.jsx';

export default function DevSignup() {
  const [email, setEmail] = useState('');
  const [status, setStatus] = useState('idle'); // idle | submitting | success | error
  const [error, setError] = useState('');

  async function onSubmit(e) {
    e.preventDefault();
    setStatus('submitting');
    setError('');
    try {
      if (!supabaseReady) {
        // Graceful no-op when Supabase keys are missing.
        await new Promise((r) => setTimeout(r, 400));
        setStatus('success');
        return;
      }
      const { error: err } = await supabase.from('dev_signups').insert({ email });
      if (err) throw err;
      setStatus('success');
    } catch (e) {
      setStatus('error');
      setError(e.message || 'Something went wrong.');
    }
  }

  return (
    <section
      id="devsignup"
      className="min-h-screen flex items-center px-8 md:px-16 max-w-6xl mx-auto"
    >
      <div className="w-full max-w-xl">
        <div className="text-sm uppercase tracking-widest text-amber font-num mb-6">
          Early Access
        </div>
        <h2 className="text-h1 tracking-tightest mb-8">
          {copy.devSignup.heading}
        </h2>
        {status === 'success' ? (
          <p className="text-lead text-ink">
            Got it. We'll be in touch from <span className="text-amber">{copy.contact.email}</span>.
          </p>
        ) : (
          <form onSubmit={onSubmit} className="flex flex-col gap-6">
            <Input
              type="email"
              required
              placeholder="you@domain.com"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              disabled={status === 'submitting'}
            />
            <div className="flex items-center gap-4">
              <Button type="submit" disabled={status === 'submitting'}>
                {status === 'submitting' ? 'Sending…' : copy.devSignup.cta}
              </Button>
              {error && <span className="text-sm text-red-400">{error}</span>}
            </div>
          </form>
        )}
      </div>
    </section>
  );
}
```

- [ ] **Step 2: Mount in Landing**

Modify `Landing.jsx` — import and render `<DevSignup />`.

- [ ] **Step 3: Commit**

```bash
git add site/
git commit -m "feat(site): add devsignup section with supabase submit + graceful fallback"
```

---

## Task 21: `Contact` section + final render of Landing

**Files:**
- Create: `site/src/sections/Contact.jsx`
- Modify: `site/src/pages/Landing.jsx`

- [ ] **Step 1: Write `Contact.jsx`**

Create `site/src/sections/Contact.jsx`:
```jsx
import { copy } from '@/content/copy.js';

export default function Contact() {
  return (
    <section
      id="contact"
      className="min-h-screen flex flex-col justify-center px-8 md:px-16 max-w-6xl mx-auto"
    >
      <div className="text-sm uppercase tracking-widest text-amber font-num mb-6">
        Contact
      </div>
      <a
        href={`mailto:${copy.contact.email}`}
        className="text-h1 tracking-tightest text-ink hover:text-amber transition-colors"
      >
        {copy.contact.email}
      </a>
      <div className="mt-10 flex gap-6 font-num text-sm text-muted">
        {copy.contact.socials.map((s) => (
          <a
            key={s.label}
            href={s.href}
            target="_blank"
            rel="noreferrer"
            className="hover:text-amber transition-colors"
          >
            {s.label} ↗
          </a>
        ))}
      </div>
      <div className="mt-16 text-xs font-num text-muted">
        © {new Date().getFullYear()} Neural Dynamics.
      </div>
    </section>
  );
}
```

- [ ] **Step 2: Overwrite `Landing.jsx` as the final composition**

Overwrite `site/src/pages/Landing.jsx`:
```jsx
import { useEffect } from 'react';
import Canvas from '@/scene/Canvas.jsx';
import Hero from '@/sections/Hero.jsx';
import Thesis from '@/sections/Thesis.jsx';
import Capabilities from '@/sections/Capabilities.jsx';
import Benchmarks from '@/sections/Benchmarks.jsx';
import DevSignup from '@/sections/DevSignup.jsx';
import Contact from '@/sections/Contact.jsx';
import useIsMobile from '@/hooks/useIsMobile.js';
import { refresh } from '@/scene/timeline.js';

export default function Landing() {
  const isMobile = useIsMobile();

  useEffect(() => {
    // Let DOM settle, then recalc ScrollTrigger offsets.
    const t = setTimeout(() => refresh(), 100);
    return () => clearTimeout(t);
  }, []);

  return (
    <>
      {!isMobile && <Canvas />}
      {isMobile && (
        <div
          aria-hidden="true"
          className="fixed inset-0 -z-10 bg-cover bg-center"
          style={{ backgroundImage: 'url(/hero-fallback.webp)' }}
        />
      )}
      <main className="relative">
        <Hero />
        <Thesis />
        <Capabilities />
        <Benchmarks />
        <DevSignup />
        <Contact />
      </main>
    </>
  );
}
```

- [ ] **Step 3: Full-page visual check**

Run `npm run dev`. Scroll through all 6 sections. Confirm:
- Hero headline reads
- Thesis paragraph reads
- Capabilities 4 items with amber index numbers
- Benchmarks 3 numbers that count up
- DevSignup form accepts an email (submit shows success even with no Supabase keys)
- Contact shows email + socials
- 3D scene visible behind everything, acts transition on scroll
- No console errors
Stop server.

- [ ] **Step 4: Commit**

```bash
git add site/
git commit -m "feat(site): compose all six sections in landing with mobile fallback"
```

---

## Task 22: Mobile fallback image

**Files:**
- Create: `site/public/hero-fallback.webp`

- [ ] **Step 1: Capture a static hero frame**

Run `npm run dev`, open `http://localhost:3000` in Chrome, resize window to **1600×1000**. With the hero at rest (0 scroll), take a screenshot (Cmd-Shift-4, select the canvas region), save as `~/Downloads/hero-fallback.png`.

- [ ] **Step 2: Convert PNG to WebP**

From `site/`:
```bash
# sips is preinstalled on macOS
sips -s format webp ~/Downloads/hero-fallback.png --out public/hero-fallback.webp
```
Verify `site/public/hero-fallback.webp` exists and its size is < 150 KB:
```bash
ls -la public/hero-fallback.webp
```

- [ ] **Step 3: Verify mobile fallback**

Run `npm run dev`. Open Chrome DevTools → toggle device toolbar → pick iPhone 14. Expected: no WebGL canvas, `hero-fallback.webp` fills the background, sections scroll normally. Stop.

- [ ] **Step 4: Commit**

```bash
git add site/public/hero-fallback.webp
git commit -m "feat(site): add mobile static hero fallback image"
```

---

## Task 23: `Blog` page (static list; real content later)

**Files:**
- Modify: `site/src/pages/Blog.jsx`

- [ ] **Step 1: Rewrite `Blog.jsx` with a minimal list layout**

Overwrite `site/src/pages/Blog.jsx`:
```jsx
import { Link } from 'react-router-dom';

// Posts live here until we decide on MDX migration (spec §10 open item).
const POSTS = [];

export default function Blog() {
  return (
    <main className="max-w-3xl mx-auto px-6 py-24">
      <Link to="/" className="text-sm uppercase tracking-widest text-muted hover:text-amber font-num">
        ← Home
      </Link>
      <h1 className="text-h1 tracking-tightest mt-6 mb-12">Writing</h1>
      {POSTS.length === 0 ? (
        <p className="text-muted text-lead">Posts land here soon.</p>
      ) : (
        <ul className="flex flex-col divide-y divide-neutral-800">
          {POSTS.map((post) => (
            <li key={post.slug} className="py-6">
              <Link to={`/blog/${post.slug}`} className="text-h2 hover:text-amber transition-colors">
                {post.title}
              </Link>
              <p className="text-muted mt-2">{post.excerpt}</p>
            </li>
          ))}
        </ul>
      )}
    </main>
  );
}
```

- [ ] **Step 2: Commit**

```bash
git add site/src/pages/Blog.jsx
git commit -m "feat(site): add minimal blog index page"
```

---

## Task 24: `ProductDetail` page

**Files:**
- Modify: `site/src/pages/ProductDetail.jsx`

- [ ] **Step 1: Rewrite `ProductDetail.jsx`**

Overwrite `site/src/pages/ProductDetail.jsx`:
```jsx
import { Link, useParams } from 'react-router-dom';

export default function ProductDetail() {
  const { productId } = useParams();
  return (
    <main className="max-w-3xl mx-auto px-6 py-24">
      <Link to="/" className="text-sm uppercase tracking-widest text-muted hover:text-amber font-num">
        ← Home
      </Link>
      <h1 className="text-h1 tracking-tightest mt-6 mb-8 uppercase">{productId}</h1>
      <p className="text-muted text-lead">
        Product detail content is authored per-product. Routes are kept so direct links from the old site still resolve.
      </p>
    </main>
  );
}
```

- [ ] **Step 2: Commit**

```bash
git add site/src/pages/ProductDetail.jsx
git commit -m "feat(site): update product-detail page layout"
```

---

## Task 25: Section smoke render tests

Each section must render without throwing when given the copy from `copy.js`.

**Files:**
- Create: `site/src/sections/Hero.test.jsx`
- Create: `site/src/sections/Thesis.test.jsx`
- Create: `site/src/sections/Capabilities.test.jsx`
- Create: `site/src/sections/Benchmarks.test.jsx`
- Create: `site/src/sections/DevSignup.test.jsx`
- Create: `site/src/sections/Contact.test.jsx`

- [ ] **Step 1: Write Hero test**

Create `site/src/sections/Hero.test.jsx`:
```jsx
import { describe, it, expect } from 'vitest';
import { render, screen } from '@testing-library/react';
import Hero from './Hero.jsx';
import { copy } from '@/content/copy.js';

describe('Hero section', () => {
  it('renders the headline and subline from copy', () => {
    render(<Hero />);
    expect(screen.getByText(copy.hero.headline)).toBeInTheDocument();
    expect(screen.getByText(copy.hero.subline)).toBeInTheDocument();
  });
});
```

- [ ] **Step 2: Write Thesis test**

Create `site/src/sections/Thesis.test.jsx`:
```jsx
import { describe, it, expect } from 'vitest';
import { render, screen } from '@testing-library/react';
import Thesis from './Thesis.jsx';
import { copy } from '@/content/copy.js';

describe('Thesis section', () => {
  it('renders the thesis body', () => {
    render(<Thesis />);
    expect(screen.getByText(copy.thesis.body)).toBeInTheDocument();
  });
});
```

- [ ] **Step 3: Write Capabilities test**

Create `site/src/sections/Capabilities.test.jsx`:
```jsx
import { describe, it, expect } from 'vitest';
import { render, screen } from '@testing-library/react';
import Capabilities from './Capabilities.jsx';
import { copy } from '@/content/copy.js';

describe('Capabilities section', () => {
  it('renders heading and every capability item', () => {
    render(<Capabilities />);
    expect(screen.getByText(copy.capabilities.heading)).toBeInTheDocument();
    for (const item of copy.capabilities.items) {
      expect(screen.getByText(item.title)).toBeInTheDocument();
    }
  });
});
```

- [ ] **Step 4: Write Benchmarks test**

Create `site/src/sections/Benchmarks.test.jsx`:
```jsx
import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen } from '@testing-library/react';
import Benchmarks from './Benchmarks.jsx';
import { copy } from '@/content/copy.js';

describe('Benchmarks section', () => {
  beforeEach(() => {
    // IntersectionObserver is not in jsdom
    class IO {
      observe() {}
      disconnect() {}
      unobserve() {}
    }
    vi.stubGlobal('IntersectionObserver', IO);
  });

  it('renders heading and all metric labels', () => {
    render(<Benchmarks />);
    expect(screen.getByText(copy.benchmarks.heading)).toBeInTheDocument();
    for (const m of copy.benchmarks.metrics) {
      expect(screen.getByText(m.label)).toBeInTheDocument();
    }
  });
});
```

- [ ] **Step 5: Write DevSignup test**

Create `site/src/sections/DevSignup.test.jsx`:
```jsx
import { describe, it, expect } from 'vitest';
import { render, screen } from '@testing-library/react';
import DevSignup from './DevSignup.jsx';
import { copy } from '@/content/copy.js';

describe('DevSignup section', () => {
  it('renders heading and cta', () => {
    render(<DevSignup />);
    expect(screen.getByText(copy.devSignup.heading)).toBeInTheDocument();
    expect(screen.getByRole('button', { name: copy.devSignup.cta })).toBeInTheDocument();
  });
});
```

- [ ] **Step 6: Write Contact test**

Create `site/src/sections/Contact.test.jsx`:
```jsx
import { describe, it, expect } from 'vitest';
import { render, screen } from '@testing-library/react';
import Contact from './Contact.jsx';
import { copy } from '@/content/copy.js';

describe('Contact section', () => {
  it('renders the email link', () => {
    render(<Contact />);
    expect(screen.getByText(copy.contact.email)).toBeInTheDocument();
  });
});
```

- [ ] **Step 7: Run all tests**

```bash
npm test
```
Expected: all section tests + all prior tests pass.

- [ ] **Step 8: Commit**

```bash
git add site/
git commit -m "test(site): add smoke render tests for all sections"
```

---

## Task 26: Playwright end-to-end scroll test

**Files:**
- Install: `@playwright/test`
- Create: `site/playwright.config.js`
- Create: `site/tests/e2e/scroll.spec.js`
- Modify: `site/package.json` (scripts already present)

- [ ] **Step 1: Install Playwright**

```bash
npm install --save-dev @playwright/test@^1.46.0
npx playwright install chromium
```

- [ ] **Step 2: Write `playwright.config.js`**

Create `site/playwright.config.js`:
```js
import { defineConfig, devices } from '@playwright/test';

export default defineConfig({
  testDir: './tests/e2e',
  timeout: 30_000,
  retries: 0,
  use: {
    baseURL: 'http://localhost:3000',
    trace: 'on-first-retry',
  },
  projects: [{ name: 'chromium', use: { ...devices['Desktop Chrome'] } }],
  webServer: {
    command: 'npm run dev',
    url: 'http://localhost:3000',
    reuseExistingServer: !process.env.CI,
    timeout: 60_000,
  },
});
```

- [ ] **Step 3: Write `tests/e2e/scroll.spec.js`**

Create `site/tests/e2e/scroll.spec.js`:
```js
import { test, expect } from '@playwright/test';

test('landing page renders all six sections in order on scroll', async ({ page }) => {
  await page.goto('/');
  const ids = ['hero', 'thesis', 'capabilities', 'benchmarks', 'devsignup', 'contact'];
  for (const id of ids) {
    const el = page.locator(`#${id}`);
    await expect(el).toHaveCount(1);
    await el.scrollIntoViewIfNeeded();
    await expect(el).toBeVisible();
  }
});

test('blog route renders', async ({ page }) => {
  await page.goto('/blog');
  await expect(page.getByRole('heading', { name: 'Writing' })).toBeVisible();
});

test('product route renders with param', async ({ page }) => {
  await page.goto('/product/x1');
  await expect(page.getByRole('heading', { name: /X1/i })).toBeVisible();
});
```

- [ ] **Step 4: Run e2e tests**

```bash
npm run test:e2e
```
Expected: 3 tests pass.

- [ ] **Step 5: Commit**

```bash
git add site/
git commit -m "test(site): add playwright e2e scroll and route tests"
```

---

## Task 27: Production build + bundle budget check

Verify the spec's perf budgets from §5.6 hold.

**Files:**
- None (observation only)

- [ ] **Step 1: Build**

```bash
npm run build
```
Expected: build succeeds, `site/dist/` written.

- [ ] **Step 2: Measure gzipped bundle sizes**

```bash
du -h dist/assets/*.js | sort -h
ls -la dist/assets | awk '{print $5, $9}' | sort -n
```
Find the `three`-chunk and the main app chunk. Gzip them:
```bash
gzip -c dist/assets/three-*.js | wc -c
gzip -c dist/assets/index-*.js | wc -c
```
Budgets (spec §5.6):
- three chunk gzipped < 400 KB (400 × 1024 = 409600 bytes)
- total JS gzipped < 600 KB (600 × 1024 = 614400 bytes)

- [ ] **Step 3: Preview and eyeball LCP**

```bash
npm run preview
```
Open `http://localhost:3000`. Open Chrome DevTools → Lighthouse → Desktop → run. Expected: Performance ≥ 90, LCP < 2.0s.

Record actual numbers in commit message below.

- [ ] **Step 4: If over budget — diagnose**

If any budget fails:
- Bundle too big → check `dist/assets/` for accidentally bundled `drei/all`. Use named imports from `@react-three/drei`.
- LCP too slow → check that font CSS is preloaded (it is, via `<link rel="preconnect">` in `index.html`).
- Do not ship if over budget. Record the overage and what was tried in the commit message, and loop back before the deploy task.

- [ ] **Step 5: Commit (observation + any tuning changes)**

```bash
git add -A
git commit -m "chore(site): production build and bundle-budget report

- three chunk gzipped: <N> KB
- total JS gzipped: <N> KB
- Lighthouse Perf (desktop): <N>
- LCP: <N>s"
```

---

## Task 28: Hostinger first deploy (manual SFTP, pre-editor)

Phase B will automate this; for Phase A we deploy once by hand to confirm SFTP path, domain, and build correctness.

**Files:**
- None

- [ ] **Step 1: Gather Hostinger SFTP credentials**

From Hostinger hPanel → Files → FTP Accounts:
- Host (e.g. `files.000webhost.com` or `ftp.neurldynamicsteam.io`)
- Port (22 for SFTP, 21 for FTP)
- Username
- Password
- Remote path (usually `public_html` or `domains/neurldynamicsteam.io/public_html`)

Confirm SFTP is available (not plain FTP). If only FTP is offered, Phase B will need to use FTP instead of SFTP — note this in an open item.

- [ ] **Step 2: Build**

```bash
npm run build
```

- [ ] **Step 3: Upload `dist/` to remote `public_html`**

Use Transmit/Cyberduck/FileZilla/`sftp` CLI. Minimum verification using `sftp` CLI:
```bash
sftp -P <port> <user>@<host>
# then, in the sftp prompt:
cd public_html
put -r dist/* .
bye
```

- [ ] **Step 4: Verify the live site**

Open `https://www.neurldynamicsteam.io` in a browser (incognito, cache-cleared). Expected:
- Hero headline matches `copy.hero.headline`
- Scroll through all 6 sections works
- 3D scene renders
- No mixed-content warnings, no 404s in Network tab
- `/blog` and `/product/x1` resolve (Hostinger should serve `index.html` for unknown paths — if not, add `.htaccess`; see next step)

- [ ] **Step 5: If client-side routes 404 — add `.htaccess`**

Create `site/public/.htaccess` with:
```
<IfModule mod_rewrite.c>
  RewriteEngine On
  RewriteBase /
  RewriteRule ^index\.html$ - [L]
  RewriteCond %{REQUEST_FILENAME} !-f
  RewriteCond %{REQUEST_FILENAME} !-d
  RewriteRule . /index.html [L]
</IfModule>
```
Rebuild and re-upload. Confirm `/blog` now works.

- [ ] **Step 6: Commit any routing fix**

```bash
git add -A
git commit -m "chore(site): add htaccess rewrite for client-side routing"
```

- [ ] **Step 7: Tag the first deploy**

```bash
git tag -a phase-a-deploy-1 -m "First manual Hostinger deploy of Phase A"
```

---

## Self-Review (performed by plan author)

**Spec coverage:**
- ✅ Tech stack (§5.1) — Tasks 1, 2, 3, 6, 7, 8
- ✅ Color & typography (§5.2) — Task 2
- ✅ Six-act scroll scene (§5.3) — Tasks 10–15
- ✅ File structure (§5.4) — enforced across tasks
- ✅ Copy data model (§5.5) — Tasks 4, 5
- ✅ Performance budgets (§5.6) — Task 27
- ✅ Pages: Landing, Blog, ProductDetail — Tasks 3, 21, 23, 24
- ✅ Mobile fallback (§5.6) — Tasks 21, 22
- ✅ Tests: copy.js shape, section smoke, e2e scroll (§8.1) — Tasks 4, 25, 26
- ✅ First Hostinger deploy (§9 Phase A end) — Task 28
- ❌ Editor — intentionally deferred to Phase B plan (out of scope here per spec §9 and scope-check at top)

**Placeholder scan:**
- No "TBD"/"TODO"/"implement later" in the body.
- `copy.js` placeholder benchmarks get a `// PLACEHOLDER` comment in Task 5 — this is an explicit open item tied to spec §10, not a plan hole.
- `tests/e2e/scroll.spec.js` hardcodes section ids — the `SECTION_IDS` export in `timeline.js` is the source of truth; ids match.

**Type consistency:**
- `copy` object shape identical across Tasks 4, 5, 16–21, 25.
- `SECTION_IDS` used in `timeline.js`, matched by `id=` attributes in section components.
- `store[sectionId]` keys in `Canvas.jsx` (`thesis`, `capabilities`, `benchmarks`, `devsignup`) match the exact ids set on each section.
- `supabaseReady` exported from `lib/supabase.js` in Task 6 and consumed in Task 20.
- `useIsMobile` exported as default in Task 7, imported as default in Task 21.
- `Number` default export in Task 6, consumed in Task 19.

No inconsistencies found.
