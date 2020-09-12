const fs = require('fs');
const cv = require('opencv4nodejs');
const FakeWebcam = require('./FakeWebcam');
const tf = require('@tensorflow/tfjs-node-gpu');
const tfn = require("@tensorflow/tfjs-node");
const bodyPix = require('@tensorflow-models/body-pix');
const {SmoothFaceFeatures, SmoothOrientation} = require('./SmoothPose');
const RenderEngine = require('./RenderEngine');
const {
    performance,
    PerformanceObserver
} = require('perf_hooks');
const columnify = require('columnify')


/**
 * Constantly and always profile performance
 */
const profiling = {};
const obs = new PerformanceObserver((list) => {
    const entry = list.getEntries()[0];
    if (profiling[entry.name]) {
        profiling[entry.name].push(entry.duration);
        if (profiling[entry.name].length > CONFIG.profiling_max_samples) {
            profiling[entry.name].shift();
        }
    } else {
        profiling[entry.name] = [entry.duration];
    }
});
obs.observe({entryTypes: ['measure'], buffered: false});


/**
 * Configurable parameters
 */
global.CONFIG = JSON.parse(fs.readFileSync('./resources/config.json', 'utf8').toString());


/**
 * The source (physical) webcam
 */
const cap = new cv.VideoCapture(CONFIG.real_webcam);
cap.set(cv.CAP_PROP_FRAME_WIDTH, CONFIG.width);
cap.set(cv.CAP_PROP_FRAME_HEIGHT, CONFIG.height);
cap.set(cv.CAP_PROP_FPS, 60);

const faceFeaturesMovingAverage = new SmoothFaceFeatures(CONFIG.moving_avg_ratio, CONFIG.width, CONFIG.height);
const orientationMovingAverage = new SmoothOrientation(CONFIG.moving_avg_ratio);
const fw = new FakeWebcam(CONFIG.width, CONFIG.height);
const renderEngine = new RenderEngine(CONFIG.width/2, CONFIG.height/2);


/**
 * The arrays for reversing the one-hot vectors to a, b, c, and z
 */
const oneHotVectorReverseMapping = {
    a_steps: [-20, -17, -14, -11, -8, -5, -2, 1, 4, 7, 10,
        13, 16, 19, 22, 25, 28],
    b_steps: [-40, -37, -34, -31, -28, -25, -22, -19, -16,
        -13, -10, -7, -4, -1, 2, 5, 8, 11, 14, 17, 20, 23,
        26, 29, 32, 35, 38],
    c_steps: [-70, -68, -66, -64, -62, -60, -58, -56, -54,
        -52, -50, -48, -46, -44, -42, -40, -38, -36, -34, -32,
        -30, -28, -26, -24, -22, -20, -18, -16, -14, -12, -10,
        -8, -6, -4, -2, 0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20,
        22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44, 46, 48,
        50, 52, 54, 56, 58, 60, 62, 64, 66, 68, 70],
    z_steps: [-9.0, -8.5, -8.0, -7.5, -7.0, -6.5, -6.0, -5.5,
        -5.0, -4.5, -4.0, -3.5, -3.0, -2.5, -2.0, -1.5, -1.0,
        -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5,
        5.0],
};

fw.init(CONFIG.fake_webcam)
    .then(async _ => {
        //performance.mark("BodyPix - loading the model-init");
        const bodyPixModel = await bodyPix.load(CONFIG.body_pix_loading_parameters);
        //performance.mark("BodyPix - loading the model-end");
        //performance.measure("BodyPix - loading the model",
        //    "BodyPix: loading the model-init",
        //    "BodyPix: loading the model-end");

        const handler = tfn.io.fileSystem(CONFIG.face_orientation_model);
        const faceOrientationModel = await tf.loadLayersModel(handler);

        let mask = new Uint8Array(CONFIG.height * CONFIG.width);
        const dilateKernel = new cv.Mat(CONFIG.dilation, CONFIG.dilation, cv.CV_8UC1, [255]);

        let background_src = cv.imread(CONFIG.background);
        let background = background_src.convertTo(cv.CV_32FC3, 1.0 / 255);
        const allOnes = new cv.Mat(CONFIG.height, CONFIG.width, cv.CV_32FC3, [1.0, 1.0, 1.0]);

        await renderEngine.init();

        //Warm up
        //performance.mark("BodyPix - warm up-init");
        {
            const backgroundData = background_src.getData();
            const bodyPixInput = tf.tensor3d(backgroundData,
                [background_src.rows, background_src.cols,
                    background_src.channels]);
            await bodyPixModel.segmentPerson(bodyPixInput, CONFIG.body_pix_segmentation_parameters);
        }
        //performance.mark("BodyPix - warm up-end");
        //performance.measure("BodyPix - warm up",
        //    "BodyPix - warm up-init",
        //    "BodyPix - warm up-end");

        while (1) {
            performance.mark("Total - each frame-init");

            performance.mark("Webcam - reading from physical-init");
            const frame = cap.read();
            performance.mark("Webcam - reading from physical-end");
            performance.measure("* Webcam - reading from physical",
                "Webcam - reading from physical-init",
                "Webcam - reading from physical-end");

            performance.mark("BodyPix - inference-init");
            const frameData = frame.getData();
            const bodyPixInput = tf.tensor3d(frameData,
                [frame.rows, frame.cols, frame.channels]);
            const segmentation = await bodyPixModel.segmentPerson(bodyPixInput,
                {
                    flipHorizontal: false,
                    internalResolution: 'full',//medium
                    segmentationThreshold: 0.7,
                });
            tf.dispose(bodyPixInput);                
            performance.mark("BodyPix - inference-end");
            performance.measure("* BodyPix - inference",
                "BodyPix - inference-init",
                "BodyPix - inference-end");

            for (let i = 0; i < segmentation.data.length; i++) {
                mask[i] = segmentation.data[i] > 0 ? 255 : 0;
            }

            async function applyMask(frame, mask, background) {
                //Prepare the mask
                performance.mark("OpenCV - masking the frame-init");
                let maskMat = new cv.Mat(Buffer.from(mask),
                    frame.rows, frame.cols, cv.CV_8UC1);
                maskMat = await maskMat.dilateAsync(dilateKernel);
                maskMat = maskMat.gaussianBlur(new cv.Size(CONFIG.blur, CONFIG.blur), 0);
                maskMat = maskMat.cvtColor(cv.COLOR_GRAY2RGB)
                    .convertTo(cv.CV_32FC3, 1.0 / 255);

                //Apply the mask
                let fg = frame.convertTo(cv.CV_32FC3, 1.0 / 255);
                fg = maskMat.hMul(fg);
                let invMask = allOnes.sub(maskMat);
                let bg = invMask.hMul(background);
                let output = fg.add(bg);
                performance.mark("OpenCV - masking the frame-end");
                performance.measure("* OpenCV - masking the frame",
                    "OpenCV - masking the frame-init",
                    "OpenCV - masking the frame-end");

                return output;
            }

            async function renderHelmet(segmentation) {
                //Draw key points
                performance.mark("Render - the whole process-init");
                const smoothedMarker = faceFeaturesMovingAverage.addSample(segmentation);
                if (smoothedMarker) {
                    const leftEye_x =
                        (smoothedMarker["leftEye"].position.x - CONFIG.width/2) / CONFIG.width;
                    const leftEye_y =
                        (smoothedMarker["leftEye"].position.y - CONFIG.height/2) / CONFIG.height;
                    const rightEye_x =
                        (smoothedMarker["rightEye"].position.x - CONFIG.width/2) / CONFIG.width;
                    const rightEye_y =
                        (smoothedMarker["rightEye"].position.y - CONFIG.height/2) / CONFIG.height;

                    const leftEar_x =
                        (smoothedMarker["leftEar"].position.x - CONFIG.width/2) / CONFIG.width;
                    const leftEar_y =
                        (smoothedMarker["leftEar"].position.y - CONFIG.height/2) / CONFIG.height;
                    const rightEar_x =
                        (smoothedMarker["rightEar"].position.x - CONFIG.width/2) / CONFIG.width;
                    const rightEar_y =
                        (smoothedMarker["rightEar"].position.y - CONFIG.height/2) / CONFIG.height;

                    const eyes_dx = leftEye_x - rightEye_x;
                    const eyes_dy = leftEye_y - rightEye_y;
                    const left_eye_ear_dx = leftEye_x - leftEar_x;
                    const left_eye_ear_dy = leftEye_y - leftEar_y;
                    const right_eye_ear_dx = rightEye_x - rightEar_x;
                    const right_eye_ear_dy = rightEye_y - rightEar_y;
                    const eyes_distance =
                        Math.sqrt(eyes_dx * eyes_dx + eyes_dy * eyes_dy);
                    const eyes_angle = Math.asin(eyes_dy / eyes_distance);
                    const left_ear_to_left_eye =
                        Math.sqrt(left_eye_ear_dx * left_eye_ear_dx +
                                left_eye_ear_dy * left_eye_ear_dy);
                    const right_ear_to_right_eye =
                        Math.sqrt(right_eye_ear_dx * right_eye_ear_dx +
                            right_eye_ear_dy * right_eye_ear_dy)

                    const example = tf.tensor2d([[
                        eyes_dx,
                        eyes_dy,
                        left_eye_ear_dx,
                        left_eye_ear_dy,
                        right_eye_ear_dx,
                        right_eye_ear_dy,
                        eyes_angle,
                        eyes_distance,
                        left_ear_to_left_eye,
                        right_ear_to_right_eye,
                    ]]);
                    const prediction = faceOrientationModel.predict(example);
                    const pred_a_out = tf.argMax(prediction[0], 1).arraySync();
                    const pred_b_out = tf.argMax(prediction[1], 1).arraySync();
                    const pred_c_out = tf.argMax(prediction[2], 1).arraySync();
                    const pred_z_out = tf.argMax(prediction[3], 1).arraySync();
                    const x = (smoothedMarker["nose"].position.x - CONFIG.width/2)
                        / CONFIG.width * CONFIG['2d_to_3d_coefficient'] + CONFIG['2d_x_offset'];
                    const y = -(smoothedMarker["nose"].position.y - CONFIG.height/2)
                        / CONFIG.height * CONFIG['2d_to_3d_coefficient'] + CONFIG['2d_y_offset'];

                    const helmet = await renderEngine.render(orientationMovingAverage.addSample({
                        a: oneHotVectorReverseMapping.a_steps[pred_a_out[0]],
                        b: oneHotVectorReverseMapping.b_steps[pred_b_out[0]],
                        c: oneHotVectorReverseMapping.c_steps[pred_c_out[0]],
                        x,
                        y,
                        z: oneHotVectorReverseMapping.z_steps[pred_z_out[0]],
                    }));
                    helmet.resize(CONFIG.width, CONFIG.height);

                    const [helmetB, helmetG, helmetR, helmetA] =
                        helmet.resize(CONFIG.height, CONFIG.width).splitChannels();
                    const helmetF32 = (new cv.Mat([helmetB, helmetG, helmetR]))
                        .convertTo(cv.CV_32FC3, 1.0 / 255);

                    const helmetMaskF32 =
                        (new cv.Mat([helmetA, helmetA, helmetA]))
                            .convertTo(cv.CV_32FC3, 1.0 / 255);
                    const invHelmetMaskF32 = allOnes.sub(helmetMaskF32);
                    const helmetMasked = helmetMaskF32.hMul(helmetF32);

                    return {
                        helmet: helmetMasked,
                        invHelmetMaskF32,
                    };
                }
                performance.mark("Render - the whole process-end");
                performance.measure("* Render - the whole process",
                    "Render - the whole process-init",
                    "Render - the whole process-end");
            }

            const threads = await Promise.all([
                applyMask(frame, mask, background),
                renderHelmet(segmentation),
            ]);


            const bgAndFrame = threads[0];
            let output = null;
            if (threads[1]) {
                const {invHelmetMaskF32, helmet} = threads[1];
                output = helmet.add(
                    invHelmetMaskF32.hMul(bgAndFrame)
                );
            } else {
                output = bgAndFrame;
            }

            performance.mark("Webcam - write to virtual-init");
            fw.write_frame(output.convertTo(cv.CV_8UC3, 255.0));
            performance.mark("Webcam - write to virtual-end");
            performance.measure("* Webcam - write to virtual",
                "Webcam - write to virtual-init",
                "Webcam - write to virtual-end");


            performance.mark("Total - each frame-end");
            performance.measure("Total - each frame",
                "Total - each frame-init",
                "Total - each frame-end");

            console.clear();
            const profiling_results = [];
            for (let name of Object.keys(profiling)) {
                const sum = profiling[name].reduce((a, b) => a + b, 0);
                const squared_sum = profiling[name]
                    .reduce((a, b) => a + b * b, 0);
                const count = profiling[name].length;
                const avg = (sum / count).toFixed(2);
                profiling_results.push({
                    name,
                    Average: avg,
                    STDEV: Math.sqrt(squared_sum / count - avg * avg)
                        .toFixed(2),
                });
            }
            console.log(columnify(profiling_results
                .sort((a, b) => a.name.localeCompare(b.name)),
                {
                    config: {
                        name: {minWidth: 20},
                    },
                }));
        }

    })
    .catch(err => {
        console.log(err)
    });

