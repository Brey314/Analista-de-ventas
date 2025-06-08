document.addEventListener('DOMContentLoaded', function() {
    // Mobile menu toggle
    const mobileMenuBtn = document.querySelector('.mobile-menu');
    const mainNav = document.querySelector('.main-nav ul');

    mobileMenuBtn.addEventListener('click', function() {
        mainNav.classList.toggle('show');
        this.querySelector('i').classList.toggle('fa-times');
        this.querySelector('i').classList.toggle('fa-bars');
    });

    // Sticky header
    const header = document.querySelector('.main-header');
    window.addEventListener('scroll', function() {
        if (window.scrollY > 100) {
            header.classList.add('scrolled');
        } else {
            header.classList.remove('scrolled');
        }
    });

    // Active nav link on scroll
    const sections = document.querySelectorAll('section');
    const navLinks = document.querySelectorAll('.nav-link');

    window.addEventListener('scroll', function() {
        let current = '';

        sections.forEach(section => {
            const sectionTop = section.offsetTop;
            const sectionHeight = section.clientHeight;

            if (pageYOffset >= sectionTop - 300) {
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

    // Animaciones con GSAP y ScrollTrigger
    gsap.registerPlugin(ScrollTrigger);

    // Animaciones generales para elementos con data-animate
    document.querySelectorAll('[data-animate]').forEach(el => {
        const animation = el.getAttribute('data-animate');
        const delay = el.getAttribute('data-delay') || 0;

        gsap.from(el, {
            scrollTrigger: {
                trigger: el,
                start: "top 80%",
                toggleActions: "play none none none"
            },
            opacity: 0,
            y: 50,
            duration: 0.8,
            delay: parseFloat(delay),
            ease: "power2.out"
        });
    });

    // Animación especial para las tarjetas de características
    gsap.utils.toArray('.feature-card').forEach((card, i) => {
        gsap.from(card, {
            scrollTrigger: {
                trigger: card,
                start: "top 80%",
                toggleActions: "play none none none"
            },
            opacity: 0,
            y: 50,
            duration: 0.6,
            delay: i * 0.1,
            ease: "back.out(1.7)"
        });
    });

    // Contador animado para las estadísticas
    const counters = document.querySelectorAll('.stat-number');

    if (counters) {
        counters.forEach(counter => {
            const target = +counter.getAttribute('data-count');
            const count = +counter.innerText;
            const duration = 2000;
            const increment = target / (duration / 16);

            const updateCount = () => {
                const current = +counter.innerText;
                if (current < target) {
                    counter.innerText = Math.ceil(current + increment);
                    setTimeout(updateCount, 16);
                } else {
                    counter.innerText = target;
                }
            };

            ScrollTrigger.create({
                trigger: counter,
                start: "top 80%",
                onEnter: updateCount,
                once: true
            });
        });
    }

    // Testimonials carousel
    const testimonials = document.querySelectorAll('.testimonial');
    const dotsContainer = document.querySelector('.carousel-dots');
    let currentTestimonial = 0;

    // Crear dots dinámicamente
    testimonials.forEach((_, index) => {
        const dot = document.createElement('div');
        dot.classList.add('carousel-dot');
        if (index === 0) dot.classList.add('active');
        dot.addEventListener('click', () => showTestimonial(index));
        dotsContainer.appendChild(dot);
    });

    function showTestimonial(index) {
        testimonials[currentTestimonial].classList.remove('active');
        dotsContainer.children[currentTestimonial].classList.remove('active');

        currentTestimonial = (index + testimonials.length) % testimonials.length;

        testimonials[currentTestimonial].classList.add('active');
        dotsContainer.children[currentTestimonial].classList.add('active');
    }

    document.querySelector('.prev-button').addEventListener('click', () => {
        showTestimonial(currentTestimonial - 1);
    });

    document.querySelector('.next-button').addEventListener('click', () => {
        showTestimonial(currentTestimonial + 1);
    });

    // Auto-rotate testimonials
    let testimonialInterval = setInterval(() => {
        showTestimonial(currentTestimonial + 1);
    }, 5000);

    // Pausar el carrusel al interactuar
    const carousel = document.querySelector('.testimonials-carousel');
    carousel.addEventListener('mouseenter', () => {
        clearInterval(testimonialInterval);
    });

    carousel.addEventListener('mouseleave', () => {
        testimonialInterval = setInterval(() => {
            showTestimonial(currentTestimonial + 1);
        }, 5000);
    });

    // Mostrar/ocultar botón flotante de WhatsApp y "volver arriba"
    const whatsappBtn = document.querySelector('.whatsapp-float');
    const backToTopBtn = document.querySelector('.back-to-top');
    const floatingCta = document.querySelector('.floating-cta');

    window.addEventListener('scroll', function() {
        if (window.scrollY > 300) {
            backToTopBtn.classList.add('visible');
            floatingCta.classList.add('visible');
        } else {
            backToTopBtn.classList.remove('visible');
            floatingCta.classList.remove('visible');
        }
    });

    // Formulario de contacto
    const contactForm = document.getElementById('contactForm');
    if (contactForm) {
        contactForm.addEventListener('submit', function(e) {
            e.preventDefault();

            // Simular envío
            const formData = new FormData(this);
            const formObject = {};
            formData.forEach((value, key) => {
                formObject[key] = value;
            });

            console.log('Formulario enviado:', formObject);

            // Mostrar mensaje de éxito
            const successMsg = document.createElement('div');
            successMsg.className = 'form-success';
            successMsg.innerHTML = `
                <i class="fas fa-check-circle"></i>
                <p>¡Gracias por tu mensaje! Nos pondremos en contacto contigo pronto.</p>
            `;

            contactForm.parentNode.insertBefore(successMsg, contactForm);
            contactForm.style.display = 'none';

            // Animación de éxito
            gsap.from(successMsg, {
                opacity: 0,
                y: 20,
                duration: 0.5
            });

            // Resetear formulario después de 5 segundos (simulado)
            setTimeout(() => {
                contactForm.reset();
                contactForm.style.display = 'block';
                successMsg.remove();
            }, 5000);
        });
    }

    // Efecto parallax para la sección de blog
    const parallaxBg = document.querySelector('.parallax-bg');
    if (parallaxBg) {
        gsap.to(parallaxBg, {
            scrollTrigger: {
                trigger: '.blog-section',
                scrub: true
            },
            y: 100,
            ease: "none"
        });
    }

    // Efecto de onda en las tarjetas de características
    const featureWaves = document.querySelectorAll('.feature-wave');
    featureWaves.forEach(wave => {
        const path = wave.querySelector('path');
        const originalD = path.getAttribute('d');

        gsap.to(path, {
            duration: 4,
            attr: { d: originalD.replace(/V(\d+)/, 'V' + (parseInt(originalD.match(/V(\d+)/)[1]) + 5)) },
            repeat: -1,
            yoyo: true,
            ease: "sine.inOut"
        });
    });

    // Animación de las burbujas en la sección "Sobre Nosotros"
    const circles = document.querySelectorAll('.circle-decor');
    circles.forEach((circle, i) => {
        gsap.to(circle, {
            duration: 10 + i * 2,
            x: i % 2 === 0 ? '+=20' : '-=20',
            y: i % 2 === 0 ? '+=10' : '-=10',
            repeat: -1,
            yoyo: true,
            ease: "sine.inOut"
        });
    });
});