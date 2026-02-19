const cacheName = 'smart-advisor-v2';
const assets = ['./', './index.html', './style.css', './script.js', './manifest.json'];

self.addEventListener('install', e => {
    e.waitUntil(caches.open(cacheName).then(cache => cache.addAll(assets)));
});

self.addEventListener('fetch', e => {
    e.respondWith(caches.match(e.request).then(res => res || fetch(e.request)));
});