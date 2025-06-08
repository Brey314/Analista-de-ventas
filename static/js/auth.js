// Manejo del formulario de registro
document.addEventListener('DOMContentLoaded', function() {
    const registerForm = document.getElementById('registerForm');
    
    if (registerForm) {
        registerForm.addEventListener('submit', function(e) {
            e.preventDefault();
            
            // Validación de contraseña
            const password = document.getElementById('password').value;
            if (password.length < 8) {
                alert('La contraseña debe tener al menos 8 caracteres');
                return;
            }
            
            // Validación de términos
            const terms = document.getElementById('terms').checked;
            if (!terms) {
                alert('Debes aceptar los términos y condiciones');
                return;
            }
            
            // Recoger datos del formulario
            const formData = new FormData(this);
            const userData = {};
            formData.forEach((value, key) => {
                userData[key] = value;
            });
            
            // Simular registro (en un caso real, aquí iría una llamada a la API)
            console.log('Usuario registrado:', userData);
            
            // Mostrar mensaje de éxito
            alert('¡Registro exitoso! Por favor revisa tu correo para confirmar tu cuenta.');
            
            // Redirigir al dashboard (simulado)
            setTimeout(() => {
                window.location.href = 'dashboard.html';
            }, 2000);
        });
        
        // Mostrar/ocultar contraseña
        const passwordInput = document.getElementById('password');
        const showPasswordToggle = document.createElement('span');
        showPasswordToggle.className = 'show-password';
        showPasswordToggle.textContent = '👁️';
        showPasswordToggle.style.cursor = 'pointer';
        showPasswordToggle.style.marginLeft = '10px';
        
        passwordInput.parentNode.appendChild(showPasswordToggle);
        
        showPasswordToggle.addEventListener('click', function() {
            if (passwordInput.type === 'password') {
                passwordInput.type = 'text';
                this.textContent = '👁️';
            } else {
                passwordInput.type = 'password';
                this.textContent = '👁️';
            }
        });
    }
    
    // Validación en tiempo real
    const passwordField = document.getElementById('password');
    if (passwordField) {
        passwordField.addEventListener('input', function() {
            const passwordFeedback = document.getElementById('password-feedback') || 
                                   document.createElement('small');
            passwordFeedback.id = 'password-feedback';
            
            if (this.value.length > 0 && this.value.length < 8) {
                passwordFeedback.textContent = 'La contraseña es demasiado corta';
                passwordFeedback.style.color = 'var(--danger-color)';
            } else if (this.value.length >= 8) {
                passwordFeedback.textContent = 'La contraseña es segura';
                passwordFeedback.style.color = 'var(--success-color)';
            } else {
                passwordFeedback.textContent = '';
            }
            
            if (!this.nextElementSibling || this.nextElementSibling.id !== 'password-feedback') {
                this.parentNode.appendChild(passwordFeedback);
            }
        });
    }
});