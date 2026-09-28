# Neural Dynamics — 3D Redesign + Prompt Editor

**Date:** 2026-04-16
**Owner:** Anil Kumar (Neural Dynamics)
**Target:** www.neurldynamicsteam.io (Hostinger)
**Source of truth:** Existing export at `~/Downloads/horizons-export-b870d9c0-abdd-4eca-b97f-879c84367732` (read-only reference; not edited in place)

---

## 1. Goal

Replace the current Horizons-built landing page with a 3D-minimalist site that communicates CIPHER's hardware story, and ship a local prompt editor that lets the owner make ongoing copy/layout edits by typing instructions in natural language and deploying them to Hostinger in one click.

## 2. Non-goals

- No server-side app for the editor (runs locally only).
- No AI-generated 3D. 3D scene is authored by hand once, then frozen in v1.
- No CMS, no auth, no backend for the site beyond the Supabase client already present.
- No mobile WebGL in v1 — static fallback below 768px.
- No ability for the editor to install dependencies, modify routes, or edit 3D scene files in v1.

## 3. Architecture

Two projects, one repo, no runtime coupling:

```
neural-dynamics/
├── site/       ← the redesigned 3D website (deployed to Hostinger)
├── editor/     ← local prompt editor (never deployed; edits site/)
└── docs/
```

`site/` builds to `site/dist/`. The editor uploads that `dist/` directory via SFTP to Hostinger's `public_html`. The editor's dependencies never ship to the site bundle.

## 4. Design decisions (locked)

| Decision | Choice | Rationale |
|---|---|---|
| Hero concept | Silicon die, macro-rotating | Tangible, hardware-coded; minimalism wants a single object |
| Accent color | Amber `#F5A524` | Differentiates from cyan/violet AI-startup default; reads as precision instrument |
| 3D scope | Scroll-driven, single continuous scene | CIPHER's story is flow (calls → intercept → route); scroll-scene is top-1% tier |
| Content scope | Consolidate 15 sections → 6 | Minimalism is about cutting |
| Editor UX | Chat panel + live preview + diff + deploy button | Fastest to ship; avoids iframe-click-scoping rabbit hole |
| Typography | Inter (UI/body), JetBrains Mono (numbers/labels) | Neutral, technical, widely supported |
| Motion library | GSAP ScrollTrigger for scene; framer-motion for small UI | GSAP handles long timelines better than framer |
| 3D library | React Three Fiber + @react-three/drei | R3F is the default |
| Mobile fallback | Static hero image, flat sections, no WebGL | Perf budget |

## 5. Site — redesign

### 5.1 Stack

- React 18 + Vite (inherited)
- React Three Fiber + `@react-three/drei`
- GSAP + ScrollTrigger
- framer-motion (already a dep; keep for UI micro-motion)
- Tailwind CSS (inherited)
- Supabase JS (inherited; used for dev signup only)

### 5.2 Color & typography

- Background: `#000000`
- Primary text: `#FAFAFA`
- Muted text: `#9A9A9A`
- Accent: `#F5A524`
- UI/body: Inter, 400/500/600 weights
- Numbers, labels, code-like: JetBrains Mono, 400/500

### 5.3 Scroll scene — six acts, one camera path

1. **Hero** — macro silicon die, close, slow rotation. Headline overlay ≤10 words. JetBrains Mono subline.
2. **Thesis** — camera pulls back; die becomes part of a GPU board. One paragraph.
3. **Capabilities** — board dissolves into floating compute blocks; each block surfaces a capability card as it passes the camera.
4. **Benchmarks** — blocks reform into 3D bar columns; numbers count up in Mono.
5. **Dev signup** — columns collapse into a single amber line (the intercept path); form to the right.
6. **Contact** — amber line fades to black; minimal footer.

All 3D runs in one `<Canvas>`. HTML sections are absolute-positioned overlays synchronized to scroll progress via ScrollTrigger. Section opacity and 3D camera keyframes live in a single choreography file (`scene/timeline.js`).

### 5.4 File structure

```
site/
├── src/
│   ├── scene/                  ← LOCKED for v1 editor
│   │   ├── Canvas.jsx          ← root R3F canvas
│   │   ├── Die.jsx             ← hero silicon die
│   │   ├── Board.jsx           ← GPU board formation
│   │   ├── ComputeBlocks.jsx   ← floating capability blocks
│   │   ├── BarColumns.jsx      ← benchmark columns
│   │   ├── InterceptPath.jsx   ← amber line
│   │   ├── timeline.js         ← scroll keyframes
│   │   └── shaders/
│   │       ├── die.vert.glsl
│   │       ├── die.frag.glsl
│   │       └── grain.frag.glsl
│   ├── sections/               ← editable (layout + Tailwind)
│   │   ├── Hero.jsx
│   │   ├── Thesis.jsx
│   │   ├── Capabilities.jsx
│   │   ├── Benchmarks.jsx
│   │   ├── DevSignup.jsx
│   │   └── Contact.jsx
│   ├── content/                ← editable (pure data)
│   │   └── copy.js             ← all headlines, paragraphs, bullets, numbers
│   ├── ui/                     ← editable (shared components)
│   │   ├── Button.jsx
│   │   ├── Input.jsx
│   │   └── Number.jsx
│   ├── pages/                  ← editable
│   │   ├── Blog.jsx
│   │   └── ProductDetail.jsx
│   ├── App.jsx
│   └── main.jsx
├── public/
│   ├── hero-fallback.webp      ← mobile static
│   └── models/                 ← .glb assets (Draco + Meshopt compressed)
├── index.html
├── vite.config.js
├── tailwind.config.js
└── package.json
```

### 5.5 Copy data model

All user-facing text lives in `src/content/copy.js` as a plain JS object with a documented shape (validated by Vitest — see §8.1):

```js
export const copy = {
  hero: {
    headline: '...',
    subline: '...',
  },
  thesis: { body: '...' },
  capabilities: {
    heading: '...',
    items: [{ title: '...', body: '...' }, ...],
  },
  benchmarks: {
    heading: '...',
    metrics: [{ label: '...', value: 0, unit: '...' }, ...],
  },
  devSignup: { heading: '...', cta: '...' },
  contact: { email: '...', socials: [...] },
}
```

90% of editor prompts resolve to edits in this file, not JSX. This minimizes regression risk from AI edits.

### 5.6 Performance budgets

| Metric | Target |
|---|---|
| LCP (fast 4G) | < 2.0s |
| 3D bundle (gzip) | < 400 KB |
| Total JS (gzip) | < 600 KB |
| Lighthouse Perf (desktop) | ≥ 90 |
| Mobile WebGL | disabled; static fallback |

Techniques: R3F tree-shaking (no `@react-three/fiber/all`), Draco + Meshopt for GLBs, lazy-load `/blog` route, preconnect to Supabase only on DevSignup in-view.

## 6. Editor — prompt-to-deploy tool

### 6.1 Stack

- Next.js 15 (App Router), TypeScript
- `@anthropic-ai/sdk` — Claude Sonnet 4.6 with prompt caching
- `ssh2-sftp-client` — Hostinger SFTP deploy
- `simple-git` — auto-commit every edit to `site/`
- Monaco diff viewer (via `@monaco-editor/react`)
- Runs on `http://localhost:4000`

### 6.2 Flow

```
You type prompt
  → /api/edit receives prompt
  → context.ts selects relevant files (regex-match prompt vs file tree)
  → claude.ts sends cached system prompt + fresh prompt + file contents
  → Claude returns JSON: [{ file, search, replace }, ...]
  → patcher.ts validates + applies edits to site/src
  → git.ts auto-commits with the prompt as message
  → preview iframe (loading http://localhost:3000) hot-reloads via vite HMR
  → diff viewer shows changes
  → You click Deploy
  → /api/build runs `vite build` in site/
  → /api/deploy uploads site/dist/ via SFTP to Hostinger public_html
  → Live in ~30s
```

### 6.3 File structure

```
editor/
├── app/
│   ├── page.tsx                ← chat + preview + diff UI
│   ├── layout.tsx
│   ├── setup/page.tsx          ← first-run credential wizard
│   └── api/
│       ├── edit/route.ts
│       ├── build/route.ts
│       ├── deploy/route.ts
│       ├── revert/route.ts
│       └── history/route.ts
├── lib/
│   ├── claude.ts               ← Anthropic SDK, prompt assembly, caching
│   ├── context.ts              ← relevant-file selection
│   ├── patcher.ts              ← SEARCH/REPLACE application
│   ├── git.ts                  ← auto-commit + revert
│   ├── sftp.ts                 ← Hostinger deploy
│   └── locked.ts               ← paths the editor is forbidden to touch
├── prompts/
│   └── system.md               ← cached system prompt (rules + file tree)
├── .env.local.example
└── package.json
```

### 6.4 Prompt contract (Claude output format)

Claude must return a JSON object exactly matching:

```json
{
  "edits": [
    { "file": "src/content/copy.js", "search": "...exact string...", "replace": "...new string..." }
  ],
  "summary": "one-line human description of the change"
}
```

SEARCH is exact-match; no regex, no fuzzy. If the file has changed since the prompt was prepared and SEARCH no longer appears, that edit fails and the chat shows "couldn't apply change to X — retry?"

### 6.5 Locked paths (editor MUST refuse)

- `site/src/scene/**`
- `site/vite.config.js`
- `site/package.json`, `site/package-lock.json`
- `site/tailwind.config.js`
- `site/src/App.jsx` (route structure)
- `site/public/models/**`

`locked.ts` contains this list; `patcher.ts` rejects any edit targeting a locked path before applying.

### 6.6 Context selection

`context.ts` receives the prompt, returns a set of files to include:

1. Always include: `site/src/content/copy.js`, file tree as text, `prompts/system.md` rules.
2. Keyword-match prompt against section names: `hero|thesis|capabilit|benchmark|signup|contact`. Each hit adds the matching `sections/X.jsx`.
3. Keyword-match against `ui/` component names.
4. Cap at 40 KB of file content total. Drop least-relevant files to fit.

Cached: system prompt + file tree (both >1024 tokens, cache-eligible).
Fresh: user prompt + file contents.

### 6.7 Credentials & setup

On first run, editor detects missing `.env.local` and opens `/setup`:
- `ANTHROPIC_API_KEY`
- `HOSTINGER_SFTP_HOST`
- `HOSTINGER_SFTP_USER`
- `HOSTINGER_SFTP_PASS`
- `HOSTINGER_SFTP_PATH` (default `public_html`)

Stored only in `editor/.env.local`, git-ignored. Never sent to Claude.

### 6.8 Safety & recovery

- Every applied edit: a git commit on `site/`'s repo, message = prompt text.
- "Revert last edit" button: `git reset --hard HEAD~1`.
- Pre-deploy guard: runs `vite build`; non-zero exit blocks deploy, error shown in UI.
- SFTP upload: to a staging dir (`public_html_staging`), then atomic rename to `public_html`. Never a half-deployed site.
- If atomic rename not supported by Hostinger's SFTP, fallback: upload with `.new` suffix, rename-move files individually after full transfer.

## 7. Error handling

| Condition | Behavior |
|---|---|
| Claude returns non-JSON or wrong schema | Chat shows error; no edits applied |
| SEARCH block not found in file | That edit skipped; user sees which failed; offered one-click retry |
| Edit targets a locked path | Rejected before apply; chat shows "`scene/` is locked in v1" |
| `vite build` fails | Deploy blocked; build stderr shown; offered `git revert` |
| SFTP connect fails | 3 retries with backoff; then failure shown |
| SFTP drops mid-upload | Staging dir left intact; next deploy resumes from clean state |
| Anthropic rate limit | Exponential backoff (3 attempts); surface error if all fail |
| User has uncommitted manual edits in `site/` | Deploy blocked with "commit or stash first" |

## 8. Testing

### 8.1 Site

- **Vitest** — `copy.js` shape validation (all required keys present), section smoke render tests (each section mounts without throwing given valid copy).
- **Playwright** — one end-to-end: load `/`, scroll from 0 to bottom, assert every section's heading is visible at the right scroll position. One load of `/blog` and one `/product/:id`.
- **3D scene** — no automated testing. Verified visually on each build.

### 8.2 Editor

- **Vitest**
  - `patcher.ts`: applies single edit, applies multi-edit, rejects locked path, rejects missing SEARCH, handles Windows line endings.
  - `context.ts`: selects right sections by keyword, respects 40 KB cap.
  - `locked.ts`: all locked patterns match intended paths.
  - `sftp.ts`: mocked transport; verifies atomic-rename logic.
- **One end-to-end**: seeded `site/` fixture + mocked Anthropic + mocked SFTP; run prompt → patch → build → deploy; assert expected file state on mock remote.

### 8.3 Not tested automatically

- Claude output quality (judgement call; eyeball in real use).
- Real Hostinger SFTP (smoke-test manually on first deploy).
- Visual regressions on 3D (manual).

## 9. Rollout

1. **Phase A — Site redesign.** Build `site/`. Manual deploy first working build to Hostinger to confirm domain + SFTP path.
2. **Phase B — Editor v1.** Build `editor/`. Test against local `site/` with mocked SFTP. Then wire real Hostinger creds.
3. **Phase C — First real edit loop.** Make 5 trivial edits via the editor (copy tweaks, color tweaks). Fix the friction points that surface.
4. **Phase D — Hand-off.** You use it daily.

Each phase is its own implementation plan.

## 10. Open items

- Exact headline text for Hero / Thesis / Capabilities / Benchmarks / Dev signup / Contact — deferred to Phase A (I'll propose during build; you approve copy).
- GPU die 3D model — sourced or authored? Decision in Phase A.
- Benchmark numbers — pulled from CIPHER session data or placeholder until real numbers land? Decision in Phase A.
- `/product/:productId` — what products are listed? Decision in Phase A.
- Blog content migration — import existing posts as MDX or start fresh? Decision in Phase A.

## 11. Out of scope for v1

- Editor cannot add new sections or new components.
- Editor cannot install npm packages.
- Editor cannot modify 3D scene or shaders.
- Editor cannot modify routes.
- No multi-user / collaborative editing.
- No rollback beyond `git reset HEAD~1` (no branching/history UI yet).
- No image generation (swapping images uses existing files in `public/`).
