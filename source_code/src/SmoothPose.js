function movingAverage(latest, prev, movingAverageRatio) {
    if (!prev) {
        return latest;
    }

    return {
        nose: {
            position: {
                x: latest.nose.position.x * movingAverageRatio + prev.nose.position.x * (1.0 - movingAverageRatio),
                y: latest.nose.position.y * movingAverageRatio + prev.nose.position.y * (1.0 - movingAverageRatio),
            }
        },
        leftEye: {
            position: {
                x: latest.leftEye.position.x * movingAverageRatio + prev.leftEye.position.x * (1.0 - movingAverageRatio),
                y: latest.leftEye.position.y * movingAverageRatio + prev.leftEye.position.y * (1.0 - movingAverageRatio),
            }
        },
        rightEye: {
            position: {
                x: latest.rightEye.position.x * movingAverageRatio + prev.rightEye.position.x * (1.0 - movingAverageRatio),
                y: latest.rightEye.position.y * movingAverageRatio + prev.rightEye.position.y * (1.0 - movingAverageRatio),
            }
        },
        leftEar: {
            position: {
                x: latest.leftEar.position.x * movingAverageRatio + prev.leftEar.position.x * (1.0 - movingAverageRatio),
                y: latest.leftEar.position.y * movingAverageRatio + prev.leftEar.position.y * (1.0 - movingAverageRatio),
            }
        },
        rightEar: {
            position: {
                x: latest.rightEar.position.x * movingAverageRatio + prev.rightEar.position.x * (1.0 - movingAverageRatio),
                y: latest.rightEar.position.y * movingAverageRatio + prev.rightEar.position.y * (1.0 - movingAverageRatio),
            }
        },
    };
}

function makeHistory(newMarker, oldMarker, width, height) {
    if (!newMarker || newMarker.position.x < 0 || newMarker.position.x >= width ||
        newMarker.position.y < 0 || newMarker.position.y >= height) {
        return oldMarker;
    }
    return newMarker;
}

class SmoothFaceFeatures
{
    constructor(movingAverageRatio, width, height) {
        this.movingAverageRatio = movingAverageRatio;
        this.width = width;
        this.height = height;
    }

    addSample(segmentation) {
        if (segmentation.allPoses && segmentation.allPoses.length > 0 && segmentation.allPoses[0].keypoints) {
            const nose = segmentation.allPoses[0].keypoints.find(kp => kp.part === 'nose');
            const leftEye = segmentation.allPoses[0].keypoints.find(kp => kp.part === 'leftEye');
            const rightEye = segmentation.allPoses[0].keypoints.find(kp => kp.part === 'rightEye');
            const leftEar = segmentation.allPoses[0].keypoints.find(kp => kp.part === 'leftEar');
            const rightEar = segmentation.allPoses[0].keypoints.find(kp => kp.part === 'rightEar');

            const marker = {
                nose: makeHistory(nose, this.marker ? this.marker.nose : undefined, this.width, this.height),
                leftEye: makeHistory(leftEye, this.marker ? this.marker.leftEye : undefined, this.width, this.height),
                rightEye: makeHistory(rightEye, this.marker ? this.marker.rightEye : undefined, this.width, this.height),
                leftEar: makeHistory(leftEar, this.marker ? this.marker.leftEar : undefined, this.width, this.height),
                rightEar: makeHistory(rightEar, this.marker ? this.marker.rightEar : undefined, this.width, this.height),
            };

            this.previousMarker = this.marker;
            this.marker = movingAverage(marker, this.previousMarker, this.movingAverageRatio);
        }
        return this.marker;
    }
}

class SmoothOrientation
{
    constructor(movingAverageRatio) {
        this.movingAverageRatio = movingAverageRatio;
    }

    addSample({a, b, c, x, y, z}) {
        if (this.marker) {
            this.previousMarker = this.marker;
            this.marker = {
                a: a * this.movingAverageRatio + this.previousMarker.a * (1.0 - this.movingAverageRatio),
                b: b * this.movingAverageRatio + this.previousMarker.b * (1.0 - this.movingAverageRatio),
                c: c * this.movingAverageRatio + this.previousMarker.c * (1.0 - this.movingAverageRatio),
                x: x * this.movingAverageRatio + this.previousMarker.x * (1.0 - this.movingAverageRatio),
                y: y * this.movingAverageRatio + this.previousMarker.y * (1.0 - this.movingAverageRatio),
                z: z * this.movingAverageRatio + this.previousMarker.z * (1.0 - this.movingAverageRatio),
            };
        }
        else {
            this.marker = {a, b, c, x, y, z};
        }
        return this.marker;
    }
}

module.exports = {SmoothFaceFeatures, SmoothOrientation};
