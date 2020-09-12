const fs = require('fs');
const ioctl = require('ioctl');
const Enum = require('enum');
const ref = require('ref');
const StructType = require('ref-struct');
const ArrayType = require('ref-array')
const Union = require('ref-union')
const cv = require('opencv4nodejs');

const v4l2_pix_format = StructType({
    width: ref.types.uint32,
    height: ref.types.uint32,
    pixelformat: ref.types.uint32,
    field: ref.types.uint32, //v4l2_field
    bytesperline: ref.types.uint32,
    sizeimage: ref.types.uint32,
    colorspace: ref.types.uint32, //v4l2_colorspace
    priv: ref.types.uint32,
});

const v4l2_rect = StructType({
    left: ref.types.int32,
    top: ref.types.int32,
    width: ref.types.int32,
    height: ref.types.int32,
});
/*
const v4l2_clip = StructType({
    c: new v4l2_rect(),
    next: ref.types.ulonglong,//ctypes.POINTER(v4l2_clip)
});
*/
const v4l2_window = StructType({
    w: v4l2_rect,
    field: ref.types.uint32,
    chromakey: ref.types.uint32,
    clips: ref.types.ulonglong, //ctypes.POINTER(v4l2_clip)
    clipcount: ref.types.uint32,
    bitmap: ref.types.ulonglong, //ctypes.c_void_p
    global_alpha: ref.types.uint8,
});

const int32Array = ArrayType(ref.types.int32);
const uint16Array = ArrayType(ref.types.uint16);
const uint32Array = ArrayType(ref.types.uint32);

const v4l2_vbi_format = StructType({
    sampling_rate: ref.types.uint32,
    offset: ref.types.uint32,
    samples_per_line: ref.types.uint32,
    sample_format: ref.types.uint32,
    start: ArrayType(ref.types.int32, 2),
    count: ArrayType(ref.types.uint32, 2),
    flags: ref.types.uint32,
    reserved: ArrayType(ref.types.uint32, 2),
});

const v4l2_sliced_vbi_format = StructType({
    service_set: ref.types.uint16,
    service_lines: ArrayType(ref.types.uint16, 2*24),
    io_size: ref.types.uint32,
    reserved: ArrayType(ref.types.uint32, 2),
});

const v4l2_format_fmt = new Union({
    pix: v4l2_pix_format,
    win: v4l2_window,
    vbi: v4l2_vbi_format,
    sliced: v4l2_sliced_vbi_format,
    raw_data: ArrayType(ref.types.uint8, 200),
});

const v4l2_format = StructType({
    type: ref.types.uint32, //v4l2_buf_type
    fmt: v4l2_format_fmt,
});

class FakeWebcam
{
    constructor(width, height) {
        this.settings = new v4l2_format();
        this.settings.type = 2;//V4L2_BUF_TYPE_VIDEO_OUTPUT
        this.settings.fmt.pix.pixelformat = 1448695129;//V4L2_PIX_FMT_YUYV
        this.settings.fmt.pix.width = width
        this.settings.fmt.pix.height = height
        this.settings.fmt.pix.field = 1;//V4L2_FIELD_NONE
        this.settings.fmt.pix.bytesperline = width * 2
        this.settings.fmt.pix.sizeimage = width * height * 2
        this.settings.fmt.pix.colorspace = 7;//V4L2_COLORSPACE_JPEG

        this.video_device = null;

        this.buffer = new Uint8Array(height * width*2);
    }

    init(devicePath) {
        const {
            O_WRONLY,
            O_SYNC
        } = fs.constants;
        const that = this;

        return new Promise((resolve, reject) => {
            fs.open(devicePath, O_WRONLY | O_SYNC, function(err, fd) {
                if (err) {
                    reject('could not open file: ' + err);
                }
                ioctl(fd, 0xc0d05605, that.settings.ref());
                that.video_device = fd;
                resolve();
            });
        });
    }

    close() {
        const that = this;
        return new Promise((resolve, reject) => {
            fs.close(that.video_device, function() {
                resolve();
            });
        });
    }

    async write_frame(frame) {
        const that = this;
        const yuv = await frame.cvtColorAsync(cv.COLOR_RGB2YUV);
        const yuvData = yuv.getData();
        const height = that.settings.fmt.pix.height;
        const width = that.settings.fmt.pix.width;

        for (let i=0; i<height; i++) {
            for (let j=0; j<width; j++) {
                that.buffer[i*width*2 + (j*2)] = yuvData[i*width*3 + j*3 + 0];
            }
            for (let j=0; j<width; j+=2) {
                that.buffer[i*width*2 + (j*2+1)] = yuvData[i*width*3 + j*3 + 2];
                that.buffer[i*width*2 + (j*2+3)] = yuvData[i*width*3 + j*3 + 1];
            }
        }

        return new Promise((resolve, reject) => {
            fs.write(that.video_device, Buffer.from(that.buffer), 0, that.buffer.length, null, function(err) {
                if (err)
                    reject('error writing file: ' + err);
                else
                    resolve();
            });
        });
    }
}

module.exports = FakeWebcam;

