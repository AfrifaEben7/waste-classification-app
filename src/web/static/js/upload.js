// Handle file input and preview
document.addEventListener('DOMContentLoaded', function() {
    const fileInput = document.getElementById('fileInput');
    const fileName = document.getElementById('fileName');
    const previewContainer = document.getElementById('preview-container');
    const imagePreview = document.getElementById('imagePreview');
    const submitBtn = document.getElementById('submitBtn');
    const uploadForm = document.getElementById('uploadForm');

    // Handle file selection
    fileInput.addEventListener('change', function(e) {
        const file = e.target.files[0];
        
        if (file) {
            // Update file name display
            fileName.textContent = file.name;
            
            // Enable submit button
            submitBtn.disabled = false;
            
            // Show image preview
            const reader = new FileReader();
            reader.onload = function(event) {
                imagePreview.src = event.target.result;
                previewContainer.style.display = 'block';
            };
            reader.readAsDataURL(file);
        } else {
            fileName.textContent = 'Choose an image file';
            submitBtn.disabled = true;
            previewContainer.style.display = 'none';
        }
    });

    // Handle form submission
    uploadForm.addEventListener('submit', function(e) {
        console.log('Form submitted!');
        submitBtn.textContent = 'Classifying...';
        submitBtn.disabled = true;
        // Don't prevent default - let form submit normally
    });

    // Drag and drop functionality
    const fileLabel = document.querySelector('.file-label');

    fileLabel.addEventListener('dragover', function(e) {
        e.preventDefault();
        fileLabel.style.borderColor = '#764ba2';
        fileLabel.style.background = '#e9ecef';
    });

    fileLabel.addEventListener('dragleave', function(e) {
        e.preventDefault();
        fileLabel.style.borderColor = '#667eea';
        fileLabel.style.background = '#f8f9fa';
    });

    fileLabel.addEventListener('drop', function(e) {
        e.preventDefault();
        fileLabel.style.borderColor = '#667eea';
        fileLabel.style.background = '#f8f9fa';
        
        const files = e.dataTransfer.files;
        if (files.length > 0) {
            fileInput.files = files;
            // Trigger change event
            const event = new Event('change', { bubbles: true });
            fileInput.dispatchEvent(event);
        }
    });
});
        <p>Confidence: ${result.confidence.toFixed(2)}%</p>
    `;
}