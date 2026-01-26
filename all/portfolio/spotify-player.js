// Random Spotify Song Player
(function() {
    // Popular songs on Spotify - using track IDs with estimated durations (in seconds)
    const spotifyTracks = [
        { id: '4cOdK2wGLETKBW3PvgPWqT', duration: 200 }, // Blinding Lights - The Weeknd
        { id: '1mea3bSkSGXuIRvnydlB5b', duration: 167 }, // As It Was - Harry Styles
        { id: '5VnDkvlS0p6h45K1dcmDy3', duration: 174 }, // Watermelon Sugar - Harry Styles
        { id: '3n3Ppam7vgaVa1iaRUc9Lp', duration: 182 }, // Someone You Loved - Lewis Capaldi
        { id: '4uLU6hMCjMI75M1A2tKUQC', duration: 215 }, // Circles - Post Malone
        { id: '7qiZfU4dY1lWllzX7mPBI3', duration: 233 }, // Shape of You - Ed Sheeran
        { id: '6f3Slt0GbA2bPZlz0aIFXN', duration: 194 }, // Bad Guy - Billie Eilish
        { id: '0VjIjW4GlUZ9YdNvs3wB5o', duration: 203 }, // Levitating - Dua Lipa
        { id: '1Je1IMUlBXcx1Fz0WE7oPT', duration: 178 }, // Good 4 U - Olivia Rodrigo
        { id: '4iV5W9uYEdYUVa79Axb7Rh', duration: 121 }, // Stay - The Kid LAROI & Justin Bieber
        { id: '2VxeLyX666F8uXCJ0dZF8B', duration: 238 }, // Heat Waves - Glass Animals
        { id: '1xznGGDReH1oQq0xzbwXaP', duration: 198 }, // Peaches - Justin Bieber
        { id: '5QO79kh1waicV47BqGRL3g', duration: 215 }, // Save Your Tears - The Weeknd
        { id: '1dGr1c8CrMLDpV6mPbImSI', duration: 179 }, // Industry Baby - Lil Nas X
        { id: '5y4XIsJb0LpWEo8O1f8yzF', duration: 137 }, // Montero - Lil Nas X
        { id: '3KkXRkHbMCARz0aVfEt68P', duration: 242 }, // Drivers License - Olivia Rodrigo
        { id: '0V3wPSX9ygBnCm8psDIegu', duration: 199 }, // Dynamite - BTS
        { id: '4uLU6hMCjMI75M1A2tKUQC', duration: 164 }, // Butter - BTS
        { id: '6habFhsOp2NvshLv26DqMb', duration: 207 }, // Shivers - Ed Sheeran
        { id: '0VjIjW4GlUZ9YdNvs3wB5o', duration: 183 }  // Don't Start Now - Dua Lipa
    ];

    const playerContainer = document.getElementById('spotifyPlayer');
    const randomBtn = document.getElementById('randomSongBtn');

    if (!playerContainer || !randomBtn) return;

    let currentTrackIndex = -1;
    let usedTracks = [];
    let autoAdvanceTimer = null;
    let currentTrackId = null;

    // Shuffle algorithm similar to Shuffle-master
    function shuffle(array) {
        const shuffled = [...array];
        for (let i = shuffled.length - 1; i > 0; i--) {
            const j = Math.floor(Math.random() * (i + 1));
            [shuffled[i], shuffled[j]] = [shuffled[j], shuffled[i]];
        }
        return shuffled;
    }

    let shuffledTracks = [];
    let currentShuffleIndex = 0;

    function getRandomTrack() {
        // If we've gone through all tracks, reshuffle
        if (currentShuffleIndex >= shuffledTracks.length || shuffledTracks.length === 0) {
            shuffledTracks = shuffle([...spotifyTracks]);
            currentShuffleIndex = 0;
        }

        const track = shuffledTracks[currentShuffleIndex];
        currentShuffleIndex++;
        currentTrackIndex = spotifyTracks.indexOf(track);
        return track;
    }

    function loadRandomSong(autoplay = true) {
        // Clear any existing auto-advance timer
        if (autoAdvanceTimer) {
            clearTimeout(autoAdvanceTimer);
            autoAdvanceTimer = null;
        }

        const track = getRandomTrack();
        const trackId = track.id;
        currentTrackId = trackId;
        
        // Clear the container first to force reload
        playerContainer.innerHTML = '';
        
        // Small delay to ensure iframe is cleared before adding new one
        setTimeout(() => {
            // Try to force autoplay - note: browser policies may still block it
            const autoplayParam = autoplay ? '&autoplay=true' : '';
            playerContainer.innerHTML = `
                <iframe 
                    id="spotifyIframe"
                    style="border-radius: 12px" 
                    src="https://open.spotify.com/embed/track/${trackId}?utm_source=generator&theme=0${autoplayParam}" 
                    width="100%" 
                    height="152" 
                    frameBorder="0" 
                    allowfullscreen="" 
                    allow="autoplay; clipboard-write; encrypted-media; fullscreen; picture-in-picture" 
                    loading="eager">
                </iframe>
            `;
            
            // Set up auto-advance timer based on track duration
            // Add a small buffer (5 seconds) to account for loading/startup time
            const durationMs = (track.duration + 5) * 1000;
            
            autoAdvanceTimer = setTimeout(() => {
                // Automatically load next song when current one should be finished
                loadRandomSong(true);
            }, durationMs);
            
            // Try to programmatically trigger play after iframe loads
            const iframe = playerContainer.querySelector('iframe');
            if (iframe && autoplay) {
                iframe.onload = function() {
                    // Try multiple times to trigger autoplay
                    setTimeout(() => {
                        try {
                            // This won't work due to cross-origin restrictions, but we try
                            const iframeDoc = iframe.contentDocument || iframe.contentWindow.document;
                            const playButton = iframeDoc?.querySelector('button[aria-label*="Play"], button[aria-label*="play"]');
                            if (playButton) {
                                playButton.click();
                            }
                        } catch (e) {
                            // Cross-origin restriction - expected
                            console.log('Autoplay attempt blocked by browser security');
                        }
                    }, 1000);
                };
            }
        }, 100);
    }

    // Wait for page to be fully loaded before initializing
    function initializePlayer() {
        // Try to load with autoplay immediately
        loadRandomSong(true);
        
        // Also try after a short delay in case first attempt fails
        setTimeout(() => {
            loadRandomSong(true);
        }, 1000);
    }

    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', initializePlayer);
    } else {
        initializePlayer();
    }

    // Also try when page becomes visible (user switches tabs back)
    document.addEventListener('visibilitychange', function() {
        if (!document.hidden) {
            setTimeout(() => {
                loadRandomSong(true);
            }, 500);
        }
    });

    // Button click handler - loads new song with autoplay
    randomBtn.addEventListener('click', function() {
        randomBtn.disabled = true;
        randomBtn.textContent = 'Loading...';
        loadRandomSong(true);
        
        // Re-enable button after a short delay
        setTimeout(() => {
            randomBtn.disabled = false;
            randomBtn.textContent = 'Play Another Song';
        }, 800);
    });

    // Listen for messages from Spotify iframe (if supported)
    window.addEventListener('message', function(event) {
        // Spotify may send messages when track ends (limited support)
        if (event.origin === 'https://open.spotify.com') {
            try {
                const data = event.data;
                // If Spotify sends track end event, load next song
                if (data && data.type === 'track-ended' || data === 'track-ended') {
                    loadRandomSong(true);
                }
            } catch (e) {
                // Ignore errors
            }
        }
    });

    // Also try to detect when iframe becomes inactive (fallback method)
    // Poll the iframe periodically to check if we should advance
    setInterval(() => {
        const iframe = document.getElementById('spotifyIframe');
        if (iframe && currentTrackId) {
            try {
                // Try to detect if track has ended (limited by CORS)
                // This is a fallback - the timer is the primary method
                const iframeSrc = iframe.src;
                if (!iframeSrc.includes(currentTrackId)) {
                    // Track changed, load next
                    loadRandomSong(true);
                }
            } catch (e) {
                // Cross-origin restriction - expected, timer will handle it
            }
        }
    }, 5000); // Check every 5 seconds as fallback
})();

