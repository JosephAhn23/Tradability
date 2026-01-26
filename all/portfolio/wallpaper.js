// Random Desktop Wallpaper System
(function() {
    const wallpaperContainer = document.getElementById('wallpaperContainer');
    if (!wallpaperContainer) return;
    
    // Desktop wallpapers from desktop-wallpapers-master folder
    const wallpapers = [
        '../desktop-wallpapers-master/1024px-Sunset_at_Varkala_Beach_Kerala_India.jpg',
        '../desktop-wallpapers-master/Beach-Wallpapers1.jpg',
        '../desktop-wallpapers-master/country-side-landscape-wallpapers-.jpg',
        '../desktop-wallpapers-master/dawn-over-beach-wallpapers_27712_1920x1200.jpg',
        '../desktop-wallpapers-master/earth-2.jpg',
        '../desktop-wallpapers-master/earth.jpg',
        '../desktop-wallpapers-master/France_Lavender_Field.jpg',
        '../desktop-wallpapers-master/globe_west_2048.jpg',
        '../desktop-wallpapers-master/Landscape_www-free-wall-paper-com-16.jpg',
        '../desktop-wallpapers-master/Landscape_www-free-wall-paper-com-23.jpg',
        '../desktop-wallpapers-master/landscape-01449.jpg',
        '../desktop-wallpapers-master/landscape-9643.jpg',
        '../desktop-wallpapers-master/landscape-national-geographic-6761345-1024-768.jpg',
        '../desktop-wallpapers-master/landscape-photo-manipulation-lost-mountain_1440x900_59913.jpg',
        '../desktop-wallpapers-master/Newport_Beach_Sunrise_California.jpg',
        '../desktop-wallpapers-master/normal_Meyrueis_-_22___23_juil_10_-_066_DxO.JPG',
        '../desktop-wallpapers-master/ocean_landscape_1280x800.jpg',
        '../desktop-wallpapers-master/space-desktop.png',
        '../desktop-wallpapers-master/Space-Rose.jpg',
        '../desktop-wallpapers-master/The-best-top-desktop-space-wallpapers-0c-hd-space-wallpaper-purple-space-nebulah.jpg',
        '../desktop-wallpapers-master/The-best-top-desktop-space-wallpapers-0d-hd-space-wallpaper-supernova.jpg',
        '../desktop-wallpapers-master/The-best-top-desktop-space-wallpapers-0l-hd-space-wallpaper-super-nova.jpg',
        '../desktop-wallpapers-master/The-best-top-desktop-space-wallpapers-0n-hd-space-wallpaper-horse-head-nebula.jpeg',
        '../desktop-wallpapers-master/The-best-top-desktop-space-wallpapers-0q-hd-space-wallpaper.jpg',
        '../desktop-wallpapers-master/The-best-top-desktop-space-wallpapers-0u-hd-space-wallpaper-pacific-ocean.jpg',
        '../desktop-wallpapers-master/The-best-top-desktop-space-wallpapers-24.jpg',
        '../desktop-wallpapers-master/The-best-top-desktop-space-wallpapers-4.jpg',
        '../desktop-wallpapers-master/The-best-top-desktop-space-wallpapers-5.png',
        '../desktop-wallpapers-master/The-best-top-desktop-space-wallpapers-9.jpg'
    ];
    
    // Get random wallpaper
    function getRandomWallpaper() {
        const randomIndex = Math.floor(Math.random() * wallpapers.length);
        return wallpapers[randomIndex];
    }
    
    // Load new wallpaper
    function loadWallpaper() {
        const wallpaperUrl = getRandomWallpaper();
        const img = new Image();
        
        img.onload = function() {
            wallpaperContainer.style.backgroundImage = `url(${wallpaperUrl})`;
            wallpaperContainer.style.opacity = '1';
        };
        
        img.onerror = function() {
            // Fallback to gradient if image fails to load
            wallpaperContainer.style.background = `
                linear-gradient(135deg, 
                    rgba(0, 20, 40, 0.9) 0%, 
                    rgba(0, 50, 100, 0.8) 50%, 
                    rgba(0, 30, 60, 0.9) 100%
                )
            `;
            wallpaperContainer.style.opacity = '1';
        };
        
        img.src = wallpaperUrl;
    }
    
    // Initial load
    loadWallpaper();
    
    // Change wallpaper every 20 seconds
    setInterval(loadWallpaper, 20000);
    
    // Change on page visibility change (when user comes back to tab)
    document.addEventListener('visibilitychange', function() {
        if (!document.hidden) {
            setTimeout(loadWallpaper, 1000);
        }
    });
})();

