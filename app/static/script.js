document.addEventListener('DOMContentLoaded', () => {
    const socket = io();

    const playing = document.querySelector("#playing");
    const looping = document.querySelector("#looping");
    const transforming = document.querySelector("#transforming");

    const buttonPressed = (element) => {
        const checked = element.checked ? "on" : "off";
        socket.emit("toggle", {"element": element.id, "state": checked});
    };
    playing.addEventListener("input", (event) => {
        buttonPressed(playing);
    });
    looping.addEventListener("input", (event) => {
        buttonPressed(looping);
    });
    transforming.addEventListener("input", (event) => {
        buttonPressed(transforming);
    });

    // Setup latent control
    const dim1 = document.querySelector("#dim1");
    const dim2 = document.querySelector("#dim2");
    const dim3 = document.querySelector("#dim3");
    const dim4 = document.querySelector("#dim4");

    const connectDim = (element) => {
        element.addEventListener("input", (event) => {
            socket.emit("dims", dim1.value, dim2.value, dim3.value, dim4.value);
        });
    };
    connectDim(dim1);
    connectDim(dim2);
    connectDim(dim3);
    connectDim(dim4);

    socket.on("set_dims", (data) => {
        console.log("received the dims: " + data);
        var values = data.dims;
        dim1.value = values[0];
        dim2.value = values[1];
        dim3.value = values[2];
        dim4.value = values[3];
    });
});
