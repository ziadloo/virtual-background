const fs = require('fs');
const puppeteer = require('puppeteer');
const cv = require('opencv4nodejs');
const tf = require('@tensorflow/tfjs-node-gpu');
const bodyPix = require('@tensorflow-models/body-pix');
const ProgressBar = require('progress');


const pup = puppeteer.launch({
	headless: true,
	args: [
		'--use-gl=swiftshader',
		'--no-sandbox',
		'--enable-surface-synchronization',
		'--disable-web-security',
	]
}).then(async browser => {
	const net = await bodyPix.load({
		architecture: 'ResNet50',
		outputStride: 16,//32
		multiplier: 1,
		quantBytes: 4,
	});

    const page = (await browser.pages())[0];
    await page.setViewport({width: 1280, height: 720});
    page.on('console', msg => console.log(msg.text()));
    page.on("pageerror", function(err) {
        const theTempValue = err.toString();
        console.log("Page error: " + theTempValue);
    });
    page.on("error", function (err) {
        const theTempValue = err.toString();
        console.log("Error: " + theTempValue);
    });

	const mats = [];
	let dataPoints = [];
	let fileIndex = 0;
	await page.exposeFunction('_setupCompleted', async () => {
		const bar = new ProgressBar('Elapsed: :elapsed, ETA: :eta, Progress: :percent (:rate frames/sec) :bar', { total: 856544 });

		for (let z=-9; z<=5; z+=0.5) {
			for (let a=-20; a<=30; a+=3) {
				for (let b=-40; b<=40; b+=3) {
					for (let c=-70; c<=70; c+=2) {
						await page.evaluate(async (a, b, c, z) => {
							model.rotation.set(Math.PI/180*a, Math.PI/180*b, Math.PI/180*c);
							model.position.set(0, 4, z);
							renderer.render(scene, camera);
							const dataUrl = renderer.domElement.toDataURL("image/png");
							await window._saveMat(a, b, c, z, dataUrl);
							delete dataUrl;
						}, a, b, c, z);
						bar.tick();
					}
				}
			}
		}

//		mats.forEach((mat, i) => cv.imwrite(`./render_output/model_${i}.png`, mat));

		fileIndex++;
		fs.writeFileSync(`./dataset_${fileIndex}.json`, JSON.stringify(dataPoints));

		console.log('all done');
		await browser.close();
	});

	await page.exposeFunction('_saveMat', async (a, b, c, z, txt) => {
		const base64Data = txt.replace(/^data:image\/png;base64,/, "");
		const buf = Buffer.from(base64Data, 'base64');
		let mat = cv.imdecode(buf, -1).cvtColor(cv.COLOR_RGBA2RGB);

tf.engine().startScope();

		const matData = mat.getData();
		const image = tf.tensor3d(matData, [mat.rows, mat.cols, mat.channels]);
		const segmentation = await net.segmentPerson(image, {
			flipHorizontal: false,
			internalResolution: 'full',//medium
			segmentationThreshold: 0.7,
		});

		if (segmentation.allPoses && segmentation.allPoses.length > 1 && segmentation.allPoses[0].keypoints) {
			const nose = segmentation.allPoses[0].keypoints.find(kp => kp.part === 'nose');
			const leftEye = segmentation.allPoses[0].keypoints.find(kp => kp.part === 'leftEye');
			const rightEye = segmentation.allPoses[0].keypoints.find(kp => kp.part === 'rightEye');
			const leftEar = segmentation.allPoses[0].keypoints.find(kp => kp.part === 'leftEar');
			const rightEar = segmentation.allPoses[0].keypoints.find(kp => kp.part === 'rightEar');

			if (nose && leftEye && rightEye && leftEar && rightEar) {
				dataPoints.push([
					nose.score, nose.position.x, nose.position.y,
					leftEye.score, leftEye.position.x, leftEye.position.y,
					rightEye.score, rightEye.position.x, rightEye.position.y,
					leftEar.score, leftEar.position.x, leftEar.position.y,
					rightEar.score, rightEar.position.x, rightEar.position.y,
					a, b, c, z,
				]);
				if (dataPoints.length % 100000 === 0) {
					fileIndex++;
					fs.writeFileSync(`./dataset_${fileIndex}.json`, JSON.stringify(dataPoints));
					dataPoints = [];
				}

//				const points = [nose, leftEye, rightEye, leftEar, rightEar];
/*
				for (let kp of points) {
					if (kp) {
					    let x = Math.round(kp.position.x);
					    let y = Math.round(kp.position.y);
					    x = x<0 ? 0 : x>1279 ? 1279 : x;
					    y = y<0 ? 0 : y>719 ? 719 : y;
						mat.set(y, x, [255, 255, 255]);
					}
				}
*/
                                if (dataPoints.length % 500 === 0) {
					global.gc();
				     //mats.push(mat);
                                     //cv.imwrite(`./render_output/model_${fileIndex}_${dataPoints.length}.png`, mat);
                                }
			}
		}

tf.engine().endScope();
		delete txt, base64Data, buf, mat, matData, image, segmentation;
	});

    await page.goto('file:///root/client-side/index.html', {
		waitUntil: 'networkidle2',
	});
});

