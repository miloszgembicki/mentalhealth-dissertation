var cacheName = "mentalhealth-v1";
var cacheFiles = [
    "index.html",
    "public/style.css",
    "public/background.png",
    "public/icon-192.png",
    "public/icon-512.png"
];

self.addEventListener("install", function (e) {
    console.log("[Service Worker] Install");
    e.waitUntil(
        caches.open(cacheName)
            .then(function (cache) {
                console.log("[Service Worker] Caching files");
                return cache.addAll(cacheFiles);
            })
            .catch(function (err) {
                console.error("[Service Worker] Cache error: " + err);
            })
    );
});

self.addEventListener("fetch", function (e) {
    e.respondWith(
        caches.match(e.request).then(function (cachedFile) {
            if (cachedFile) {
                console.log(
                    "[Service Worker] Resource fetched from the cache for: " + e.request.url
                );
                return cachedFile;
            } else {
                console.log(
                    "[Service Worker] Resource not found in cache. Fetching from network: " +
                    e.request.url
                );
                return fetch(e.request).then(function (response) {
                    if (!response || response.status !== 200 || response.type !== "basic") {
                        return response;
                    }
                    var responseToCache = response.clone();
                    caches.open(cacheName).then(function (cache) {
                        cache.put(e.request, responseToCache);
                    });
                    return response;
                });
            }
        })
    );
});