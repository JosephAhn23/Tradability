// Shuffle Master - YouTube Random Song Player
(function() {
    // Popular YouTube video IDs for music
    const youtubeVideos = [
        { id: '4NRXx6U8ABQ', duration: 200 }, // Blinding Lights - The Weeknd
        { id: 'H5v3kku4y6Q', duration: 167 }, // As It Was - Harry Styles
        { id: 'E07s5ZYygMg', duration: 174 }, // Watermelon Sugar - Harry Styles
        { id: 'zABLecsR5UE', duration: 182 }, // Someone You Loved - Lewis Capaldi
        { id: 'wXhTHyIgQ_U', duration: 215 }, // Circles - Post Malone
        { id: 'JGwWNGJdvx8', duration: 233 }, // Shape of You - Ed Sheeran
        { id: 'DyDfgMOUjCI', duration: 194 }, // Bad Guy - Billie Eilish
        { id: 'TUVcZfQe-Kw', duration: 203 }, // Levitating - Dua Lipa
        { id: 'gNi_6U5Pm_o', duration: 178 }, // Good 4 U - Olivia Rodrigo
        { id: 'kTJczQocZq8', duration: 121 }, // Stay - The Kid LAROI & Justin Bieber
        { id: 'mRD0-GxqHVo', duration: 238 }, // Heat Waves - Glass Animals
        { id: 'peBYe9v3FZY', duration: 198 }, // Peaches - Justin Bieber
        { id: 'XXYlFuWEuKI', duration: 215 }, // Save Your Tears - The Weeknd
        { id: '6ONRf7h3Mdk', duration: 179 }, // Industry Baby - Lil Nas X
        { id: '6swmTBVI83k', duration: 137 }, // Montero - Lil Nas X
        { id: 'ZmDBbnmKpqQ', duration: 242 }, // Drivers License - Olivia Rodrigo
        { id: 'gdZLi9oWNZg', duration: 199 }, // Dynamite - BTS
        { id: 'WMweEpGlu_U', duration: 164 }, // Butter - BTS
        { id: 'Il-an3ak9fg', duration: 207 }, // Shivers - Ed Sheeran
        { id: 'oygrmJFKYZY', duration: 183 }  // Don't Start Now - Dua Lipa
    ];

    const playerContainer = document.getElementById('youtubePlayer');
    const randomBtn = document.getElementById('randomSongBtn');

    if (!playerContainer || !randomBtn) return;

    let youtubePlayer = null;
    let currentVideoIndex = -1;
    let autoAdvanceTimer = null;
    let currentVideoId = null;
    let shuffledVideos = [];
    let currentShuffleIndex = 0;

    // Shuffle algorithm similar to Shuffle-master
    function shuffle(array) {
        const shuffled = [...array];
        for (let i = shuffled.length - 1; i > 0; i--) {
            const j = Math.floor(Math.random() * (i + 1));
            [shuffled[i], shuffled[j]] = [shuffled[j], shuffled[i]];
        }
        return shuffled;
    }

    function getRandomVideo() {
        // If we've gone through all videos, reshuffle
        if (currentShuffleIndex >= shuffledVideos.length || shuffledVideos.length === 0) {
            shuffledVideos = shuffle([...youtubeVideos]);
            currentShuffleIndex = 0;
        }

        const video = shuffledVideos[currentShuffleIndex];
        currentShuffleIndex++;
        currentVideoIndex = youtubeVideos.indexOf(video);
        return video;
    }

    // Wait for YouTube API to load
    function onYouTubeIframeAPIReady() {
        if (typeof YT === 'undefined' || typeof YT.Player === 'undefined') {
            setTimeout(onYouTubeIframeAPIReady, 100);
            return;
        }

        // Create YouTube player
        youtubePlayer = new YT.Player('youtubePlayer', {
            height: '152',
            width: '100%',
            playerVars: {
                'autoplay': 1,
                'controls': 1,
                'rel': 0,
                'modestbranding': 1,
                'playsinline': 1
            },
            events: {
                'onReady': onPlayerReady,
                'onStateChange': onPlayerStateChange
            }
        });
    }

    function onPlayerReady(event) {
        // Load first random video
        loadRandomVideo(true);
    }

    function onPlayerStateChange(event) {
        // When video ends (state 0), load next video
        if (event.data === YT.PlayerState.ENDED) {
            loadRandomVideo(true);
        }
    }

    function loadRandomVideo(autoplay = true) {
        // Clear any existing auto-advance timer
        if (autoAdvanceTimer) {
            clearTimeout(autoAdvanceTimer);
            autoAdvanceTimer = null;
        }

        const video = getRandomVideo();
        const videoId = video.id;
        currentVideoId = videoId;

        if (youtubePlayer) {
            // Load the video
            if (autoplay) {
                youtubePlayer.loadVideoById(videoId, 0, 'small');
            } else {
                youtubePlayer.cueVideoById(videoId, 0, 'small');
            }

            // Set up auto-advance timer as backup (in case event doesn't fire)
            const durationMs = (video.duration + 5) * 1000;
            autoAdvanceTimer = setTimeout(() => {
                loadRandomVideo(true);
            }, durationMs);
        } else {
            // If player not ready, try again
            setTimeout(() => loadRandomVideo(autoplay), 500);
        }
    }

    // Initialize when YouTube API is ready
    if (typeof YT !== 'undefined' && typeof YT.Player !== 'undefined') {
        onYouTubeIframeAPIReady();
    } else {
        // Wait for API to load
        window.onYouTubeIframeAPIReady = onYouTubeIframeAPIReady;
    }

    // Button click handler - loads new video with autoplay
    randomBtn.addEventListener('click', function() {
        randomBtn.disabled = true;
        randomBtn.textContent = 'Loading...';
        loadRandomVideo(true);
        
        // Re-enable button after a short delay
        setTimeout(() => {
            randomBtn.disabled = false;
            randomBtn.textContent = 'Play Another Song';
        }, 800);
    });

    // Also try when page becomes visible (user switches tabs back)
    document.addEventListener('visibilitychange', function() {
        if (!document.hidden && youtubePlayer) {
            setTimeout(() => {
                loadRandomVideo(true);
            }, 500);
        }
    });
})();

