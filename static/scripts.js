document.addEventListener('DOMContentLoaded', function() {
    const zoomContainer = document.querySelector('.zoom-container');
    const zoomImage = zoomContainer.querySelector('img');

    zoomContainer.addEventListener('mousedown', (e) => {
        e.preventDefault();
        e.stopPropagation();
        const rect = zoomImage.getBoundingClientRect();
        const offsetX = e.clientX - rect.left;
        const offsetY = e.clientY - rect.top;
        if (zoomImage.style.transform === 'scale(1)') {
            zoomImage.style.transform = 'scale(3)'; // Increase zoom level to 300%
            zoomImage.style.transformOrigin = `${offsetX}px ${offsetY}px`;
        } else {
            zoomImage.style.transform = 'scale(1)';
            zoomImage.style.transformOrigin = 'center center';
        }
    });

    // Prevent default drag behavior on the image
    zoomImage.addEventListener('dragstart', (e) => {
        e.preventDefault();
    });
});