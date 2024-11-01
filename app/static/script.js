document.addEventListener('DOMContentLoaded', () => {
    const socket = io();

    const playing = document.querySelector("#playing");
    const looping = document.querySelector("#looping");
    const transforming = document.querySelector("#transforming");

    const gathering = document.querySelector("#gathering");
    const training = document.querySelector("#training");
    const controlling = document.querySelector("#controlling");

    const status_info = document.querySelector("#status");
    const log_info = document.querySelector("#log");

    const xy_gathering = document.querySelector("#xy-gathering");
    const xy_control = document.querySelector("#xy-control");

    const switchControlOff = (element, state) => {
        element.checked = false;
    };
    const switchOtherControlsOff = (element) => {
        if (element != gathering)
            switchControlOff(gathering);
        if (element != training)
            switchControlOff(training);
        if (element != controlling)
            switchControlOff(controlling);
    };

    const setConsoleStatus = (element) => {
        var prefix = "panel status: ";
        if (element.checked) {
            switchOtherControlsOff(element);
            status_info.textContent = prefix + element.id;
        }
        else {
            status_info.textContent = prefix + "off";
        }
        log_info.textContent = "";
    };
    const buttonPressed = (element) => {
        const checked = element.checked ? "on" : "off";
        socket.emit("toggle", {"element": element.id, "state": checked});
    };
    const controlButtonPressed = (element) => {
        buttonPressed(element);
        setConsoleStatus(element);
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

    gathering.addEventListener("input", (event) => {
        controlButtonPressed(gathering);
        xy_gathering.classList.remove("inactive");
        xy_control.classList.add("inactive");
    });
    training.addEventListener("input", (event) => {
        controlButtonPressed(training);
    });
    controlling.addEventListener("input", (event) => {
        controlButtonPressed(controlling);
        xy_gathering.classList.add("inactive");
        xy_control.classList.remove("inactive");
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

    xy_gathering.addEventListener("click", (event) => {
        socket.emit("add_data_point", {
                        "xy": [event.clientX, event.clientY],
                        "dims": [dim1.value, dim2.value, dim3.value, dim4.value]
                    });
    });

    var event_counter = 0;
    xy_control.addEventListener("mousemove", (event) => {
        event_counter++;
        if (event_counter % 20 == 0) {
            console.log("mousemove: (" + event.clientX + ", " + event.clientY + ")");
            socket.emit("add_data_point", {
                            "xy": [event.clientX, event.clientY],
                            "dims": [dim1.value, dim2.value, dim3.value, dim4.value]
                        });
            event_counter = 0;
        }
    });

    socket.on("log", (msg) => {
        log_info.textContent = msg;
    });
    socket.on("set_dims", (data) => {
        console.log("received the dims: " + data);
        var values = data.dims;
        dim1.value = values[0];
        dim2.value = values[1];
        dim3.value = values[2];
        dim4.value = values[3];
    });
});
