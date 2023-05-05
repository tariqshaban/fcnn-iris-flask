const url = document.currentScript.getAttribute('remote');

function onSuccess(response) {
    $('#response').html(response['prediction']);

    $('#http-request-spinner').hide();
    $('#success-alert').slideDown().fadeIn();

    $('#success-alert').fadeTo(2000, 500).slideUp(500, function() {
        $('#success-alert').slideUp(500);
    });
}

function onError() {
    $('#http-request-spinner').hide();
    $('#danger-alert').slideDown().fadeIn();

    $('#danger-alert').fadeTo(2000, 500).slideUp(500, function() {
        $('#danger-alert').slideUp(500);
    });
}

async function getIrisPrediction() {
    if (!$("#form")[0].checkValidity()) {
        $("#form")[0].reportValidity();
        return;
    }

    $('#http-request-spinner').show();

    fetch(url + '?' + new URLSearchParams({
            sepalLength: document.getElementById('sepal-length').value,
            sepalWidth: document.getElementById('sepal-width').value,
            petalLength: document.getElementById('petal-length').value,
            petalWidth: document.getElementById('petal-width').value
        }), {
            method: 'get',
            headers: {
                'Accept': 'application/json',
                'Content-Type': 'application/json'
            },
        })
        .then((response) => {
            response.json().then(onSuccess)
                .catch(error => onError())
        })
        .catch(error => onError());
}

document.addEventListener("DOMContentLoaded", () => {
    $('#danger-alert').hide();
    $('#success-alert').hide();
});
