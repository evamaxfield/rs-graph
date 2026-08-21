// Prefixes an absolute internal path with the site's configured base path
// (astro.config.mjs sets base: '/rs-graph' for GitHub Pages) so internal
// links resolve correctly both in dev and once deployed.
export function withBase(path: string): string {
  const base = import.meta.env.BASE_URL.replace(/\/$/, '');
  return `${base}${path}`;
}
