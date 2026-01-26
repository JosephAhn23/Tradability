// Smooth scroll for navigation links
document.querySelectorAll('a[href^="#"]').forEach(anchor => {
    anchor.addEventListener('click', function (e) {
        e.preventDefault();
        const target = document.querySelector(this.getAttribute('href'));
        if (target) {
            const offset = 80; // Account for fixed nav
            const targetPosition = target.offsetTop - offset;
            window.scrollTo({
                top: targetPosition,
                behavior: 'smooth'
            });
        }
    });
});

// Add scroll effect to navigation
let lastScroll = 0;
const nav = document.querySelector('.nav');

window.addEventListener('scroll', () => {
    const currentScroll = window.pageYOffset;
    
    if (currentScroll > 100) {
        nav.style.background = 'rgba(10, 10, 15, 0.95)';
        nav.style.boxShadow = '0 4px 16px rgba(0, 0, 0, 0.3), 0 0 20px rgba(0, 217, 255, 0.1)';
    } else {
        nav.style.background = 'rgba(10, 10, 15, 0.95)';
        nav.style.boxShadow = '0 0 20px rgba(0, 217, 255, 0.1)';
    }
    
    lastScroll = currentScroll;
});

// Intersection Observer for fade-in animations
const observerOptions = {
    threshold: 0.1,
    rootMargin: '0px 0px -50px 0px'
};

const observer = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
        if (entry.isIntersecting) {
            entry.target.style.opacity = '1';
            entry.target.style.transform = 'translateY(0)';
        }
    });
}, observerOptions);

// Observe project cards
document.querySelectorAll('.project-card').forEach((card, index) => {
    card.style.opacity = '0';
    card.style.transform = 'translateY(30px)';
    card.style.transition = `opacity 0.6s ease ${index * 0.1}s, transform 0.6s ease ${index * 0.1}s`;
    observer.observe(card);
});

// Add parallax effect to hero section
window.addEventListener('scroll', () => {
    const scrolled = window.pageYOffset;
    const hero = document.querySelector('.hero');
    if (hero && scrolled < window.innerHeight) {
        hero.style.transform = `translateY(${scrolled * 0.5}px)`;
        hero.style.opacity = 1 - (scrolled / window.innerHeight) * 0.5;
    }
});

// Duplicate cards for seamless infinite loop
window.addEventListener('DOMContentLoaded', () => {
    // Projects wheel
    const originalWheel = document.getElementById('projectsWheel');
    const duplicateWheel = document.getElementById('projectsWheelDuplicate');
    
    if (originalWheel && duplicateWheel) {
        duplicateWheel.innerHTML = originalWheel.innerHTML;
    }
    
    // Experience wheel - only duplicate if we want infinite scroll
    // For now, let's not duplicate to avoid the double row issue
    // const experienceWheel = document.getElementById('experienceWheel');
    // const experienceWheelDuplicate = document.getElementById('experienceWheelDuplicate');
    // 
    // if (experienceWheel && experienceWheelDuplicate) {
    //     experienceWheelDuplicate.innerHTML = experienceWheel.innerHTML;
    // }
    
    // Initialize drag to scroll for all carousels
    initDragToScroll('.projects-grid');
    initDragToScroll('.experience-grid');
});

// Horizontal drag to scroll functionality
function initDragToScroll(selector) {
    const grid = document.querySelector(selector);
    if (!grid) return;
    
    let isDown = false;
    let startX;
    let scrollLeft;
    
    // Determine which type of wheel we're working with
    const isProjects = selector.includes('projects');
    const wheelSelector = isProjects ? '.projects-wheel' : '.experience-wheel';
    const cardSelector = isProjects ? '.project-card:hover' : '.experience-item:hover';
    
    // Ensure we can scroll to the beginning on load
    setTimeout(() => {
        grid.scrollLeft = 0;
    }, 100);
    
    grid.addEventListener('mousedown', (e) => {
        isDown = true;
        grid.style.cursor = 'grabbing';
        startX = e.pageX - grid.offsetLeft;
        scrollLeft = grid.scrollLeft;
        
        // Pause animation while dragging
        const wheels = grid.querySelectorAll(wheelSelector);
        wheels.forEach(wheel => {
            wheel.style.animationPlayState = 'paused';
        });
    });
    
    grid.addEventListener('mouseleave', () => {
        isDown = false;
        grid.style.cursor = 'grab';
        
        // Resume animation if not hovering over a card
        if (!document.querySelector(cardSelector)) {
            const wheels = grid.querySelectorAll(wheelSelector);
            wheels.forEach(wheel => {
                wheel.style.animationPlayState = 'running';
            });
        }
    });
    
    grid.addEventListener('mouseup', () => {
        isDown = false;
        grid.style.cursor = 'grab';
        
        // Resume animation if not hovering over a card
        if (!document.querySelector(cardSelector)) {
            const wheels = grid.querySelectorAll(wheelSelector);
            wheels.forEach(wheel => {
                wheel.style.animationPlayState = 'running';
            });
        }
    });
    
    grid.addEventListener('mousemove', (e) => {
        if (!isDown) return;
        e.preventDefault();
        const x = e.pageX - grid.offsetLeft;
        const walk = (x - startX) * 2; // Scroll speed multiplier
        
        // Calculate new scroll position
        const newScrollLeft = scrollLeft - walk;
        
        // Ensure we can scroll all the way to 0 (beginning) and to max (end)
        const maxScroll = grid.scrollWidth - grid.clientWidth;
        grid.scrollLeft = Math.max(0, Math.min(newScrollLeft, maxScroll));
    });
    
    // Touch support for mobile
    let touchStartX = 0;
    let touchScrollLeft = 0;
    
    grid.addEventListener('touchstart', (e) => {
        touchStartX = e.touches[0].pageX - grid.offsetLeft;
        touchScrollLeft = grid.scrollLeft;
        
        // Pause animation while dragging
        const wheels = grid.querySelectorAll(wheelSelector);
        wheels.forEach(wheel => {
            wheel.style.animationPlayState = 'paused';
        });
    });
    
    grid.addEventListener('touchmove', (e) => {
        if (!touchStartX) return;
        e.preventDefault();
        const x = e.touches[0].pageX - grid.offsetLeft;
        const walk = (x - touchStartX) * 2;
        
        // Calculate new scroll position
        const newScrollLeft = touchScrollLeft - walk;
        
        // Ensure we can scroll all the way to 0 (beginning) and to max (end)
        const maxScroll = grid.scrollWidth - grid.clientWidth;
        grid.scrollLeft = Math.max(0, Math.min(newScrollLeft, maxScroll));
    });
    
    grid.addEventListener('touchend', () => {
        touchStartX = 0;
        
        // Resume animation if not hovering over a card
        if (!document.querySelector(cardSelector)) {
            const wheels = grid.querySelectorAll(wheelSelector);
            wheels.forEach(wheel => {
                wheel.style.animationPlayState = 'running';
            });
        }
    });
    
    // Also support mouse wheel scrolling
    grid.addEventListener('wheel', (e) => {
        e.preventDefault();
        grid.scrollLeft += e.deltaY;
        
        // Ensure we can scroll all the way to 0 (beginning) and to max (end)
        const maxScroll = grid.scrollWidth - grid.clientWidth;
        grid.scrollLeft = Math.max(0, Math.min(grid.scrollLeft, maxScroll));
    });
}

// Back to Top Button
(function() {
    const backToTopBtn = document.getElementById('backToTopBtn');
    if (!backToTopBtn) return;

    // Button is always visible now, but you can still track scroll if needed
    // window.addEventListener('scroll', function() {
    //     if (window.pageYOffset > 300) {
    //         backToTopBtn.classList.add('show');
    //     } else {
    //         backToTopBtn.classList.remove('show');
    //     }
    // });

    // Scroll to top when clicked
    backToTopBtn.addEventListener('click', function() {
        window.scrollTo({
            top: 0,
            behavior: 'smooth'
        });
    });
})();

// Click to enlarge functionality for project cards
document.querySelectorAll('.project-card').forEach(card => {
    // Prevent click event from propagating to card when clicking View Project button
    const viewBtn = card.querySelector('.view-project-btn');
    if (viewBtn) {
        viewBtn.addEventListener('click', (e) => {
            e.stopPropagation();
        });
    }
    
    // Click to enlarge
    card.addEventListener('click', function(e) {
        // Don't enlarge if clicking the view button
        if (e.target.closest('.view-project-btn')) {
            return;
        }
        
        // Toggle enlarged state
        const isEnlarged = this.classList.contains('enlarged');
        
        // Remove enlarged from all cards
        document.querySelectorAll('.project-card').forEach(c => {
            c.classList.remove('enlarged');
        });
        
        // Toggle this card
        if (!isEnlarged) {
            this.classList.add('enlarged');
            // Pause animation when enlarged
            const grid = document.querySelector('.projects-grid');
            if (grid) {
                const wheels = grid.querySelectorAll('.projects-wheel');
                wheels.forEach(wheel => {
                    wheel.style.animationPlayState = 'paused';
                });
            }
        }
    });
    
    // Close enlarged card when clicking outside
    document.addEventListener('click', function(e) {
        const enlargedCard = document.querySelector('.project-card.enlarged');
        if (enlargedCard && !enlargedCard.contains(e.target)) {
            enlargedCard.classList.remove('enlarged');
            // Resume animation
            const grid = document.querySelector('.projects-grid');
            if (grid && !document.querySelector('.project-card:hover')) {
                const wheels = grid.querySelectorAll('.projects-wheel');
                wheels.forEach(wheel => {
                    wheel.style.animationPlayState = 'running';
                });
            }
        }
    });
    
    // Pause on hover and resume on leave
    card.addEventListener('mouseenter', function() {
        const grid = document.querySelector('.projects-grid');
        if (grid && !this.classList.contains('enlarged')) {
            grid.style.setProperty('--pause-animation', 'paused');
            const wheels = grid.querySelectorAll('.projects-wheel');
            wheels.forEach(wheel => {
                wheel.style.animationPlayState = 'paused';
            });
        }
    });
    
    card.addEventListener('mouseleave', function() {
        const grid = document.querySelector('.projects-grid');
        if (grid && !document.querySelector('.project-card:hover') && !this.classList.contains('enlarged')) {
            grid.style.setProperty('--pause-animation', 'running');
            const wheels = grid.querySelectorAll('.projects-wheel');
            wheels.forEach(wheel => {
                wheel.style.animationPlayState = 'running';
            });
        }
    });
});

// Click to enlarge functionality for experience items
document.querySelectorAll('.experience-item').forEach(item => {
    item.addEventListener('click', function(e) {
        const isEnlarged = this.classList.contains('enlarged');
        
        // Remove enlarged from all items
        document.querySelectorAll('.experience-item').forEach(i => {
            i.classList.remove('enlarged');
        });
        
        // Toggle this item
        if (!isEnlarged) {
            this.classList.add('enlarged');
            // Pause animation when enlarged
            const grid = document.querySelector('.experience-grid');
            if (grid) {
                const wheels = grid.querySelectorAll('.experience-wheel');
                wheels.forEach(wheel => {
                    wheel.style.animationPlayState = 'paused';
                });
            }
        }
    });
    
    // Pause on hover
    item.addEventListener('mouseenter', function() {
        const grid = document.querySelector('.experience-grid');
        if (grid && !this.classList.contains('enlarged')) {
            grid.style.setProperty('--pause-animation', 'paused');
            const wheels = grid.querySelectorAll('.experience-wheel');
            wheels.forEach(wheel => {
                wheel.style.animationPlayState = 'paused';
            });
        }
    });
    
    item.addEventListener('mouseleave', function() {
        const grid = document.querySelector('.experience-grid');
        if (grid && !document.querySelector('.experience-item:hover') && !this.classList.contains('enlarged')) {
            grid.style.setProperty('--pause-animation', 'running');
            const wheels = grid.querySelectorAll('.experience-wheel');
            wheels.forEach(wheel => {
                wheel.style.animationPlayState = 'running';
            });
        }
    });
});

// Close enlarged experience when clicking outside
document.addEventListener('click', function(e) {
    const enlargedExp = document.querySelector('.experience-item.enlarged');
    if (enlargedExp && !enlargedExp.contains(e.target)) {
        enlargedExp.classList.remove('enlarged');
        // Resume animation
        const grid = document.querySelector('.experience-grid');
        if (grid && !document.querySelector('.experience-item:hover')) {
            const wheels = grid.querySelectorAll('.experience-wheel');
            wheels.forEach(wheel => {
                wheel.style.animationPlayState = 'running';
            });
        }
    }
});

// Add active state to navigation links
const sections = document.querySelectorAll('section[id]');
const navLinks = document.querySelectorAll('.nav-links a');

window.addEventListener('scroll', () => {
    let current = '';
    
    sections.forEach(section => {
        const sectionTop = section.offsetTop;
        const sectionHeight = section.clientHeight;
        if (window.pageYOffset >= sectionTop - 200) {
            current = section.getAttribute('id');
        }
    });
    
    navLinks.forEach(link => {
        link.classList.remove('active');
        if (link.getAttribute('href') === `#${current}`) {
            link.classList.add('active');
        }
    });
});

// Add typing effect to hero title (optional enhancement)
const heroTitle = document.querySelector('.hero-title');
if (heroTitle) {
    const text = heroTitle.textContent;
    heroTitle.textContent = '';
    heroTitle.style.opacity = '1';
}

console.log('Portfolio loaded successfully! 🚀');
