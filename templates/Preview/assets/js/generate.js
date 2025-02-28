document.getElementById('translateForm').addEventListener('submit', function(event) {
    event.preventDefault();
    var formData = new FormData(this);
    fetch('/translate', {
        method: 'POST',
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        document.getElementById('prediction').textContent = data.prediction;

        var videoPath = '{{ url_for("serve_output_file", filename="final_output.mp4") }}';
        var videoSource = document.getElementById('videoSource');
        videoSource.src = videoPath;
        document.getElementById('outputVideo').load();
    })
    .catch(error => {
        console.error('Error:', error);
    });
});