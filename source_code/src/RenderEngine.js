const fs = require('fs');
const path = require('path');
const puppeteer = require('puppeteer');
const cv = require('opencv4nodejs');
const {performance} = require('perf_hooks');

class RenderEngine
{
    constructor(width, height) {
        this.width = width;
        this.height = height;
    }

    init() {
        const that = this;

        return new Promise(resolve => {
            const pup = puppeteer.launch({
                headless: true,
                args: [
                    '--use-gl=swiftshader',
                    '--no-sandbox',
                    '--enable-surface-synchronization',
                    '--disable-web-security',
                ]
            }).then(async browser => {
                that.browser = browser;

                that.page = (await browser.pages())[0];
                await that.page.setViewport({width: 1280, height: 720});
                that.page.on('console', msg => console.log(msg.text()));
                that.page.on("pageerror", function(err) {
                    const theTempValue = err.toString();
                    console.log("Page error: " + theTempValue);
                });
                that.page.on("error", function (err) {
                    const theTempValue = err.toString();
                    console.log("Error: " + theTempValue);
                });

                await that.page.goto(`file://${path.join(__dirname, '..')}/resources/index.html`, {
                    waitUntil: 'networkidle2',
                });

                await that.page.evaluate((width, height) => {
                    return init(width, height);
                }, that.width, that.height);

                await that.page.exposeFunction('mark', (name) => {
                    performance.mark(name);
                });

                await that.page.exposeFunction('measure', (name, start, end) => {
                    performance.measure(name, start, end);
                });

                resolve();
            });
        });
    }

    async render({a, b, c, x, y, z}) {
        performance.mark("Render - browser context-init");
        const txt = await this.page.evaluate(async (a, b, c, x, y, z) => {
            window.mark('Render - three context-init');
            model.rotation.set(Math.PI/180*a, Math.PI/180*b, Math.PI/180*c);
            model.position.set(x, y, z);
            renderer.render(scene, camera);
            window.mark('Render - three context-end');
            window.measure('* Render - three context',
                'Render - three context-init',
                'Render - three context-end');

            window.mark('Render - reading the canvas-init');
            const txt = canvas.toDataURL("image/png");
            window.mark('Render - reading the canvas-end');
            window.measure('* Render - reading the canvas',
                'Render - reading the canvas-init',
                'Render - reading the canvas-end');

            return txt;
        }, a, b, c, x, y, z);
        performance.mark("Render - browser context-end");
        performance.measure("* Render - browser context",
            "Render - browser context-init",
            "Render - browser context-end");

        performance.mark("Render - node context-init");
        const data = txt.replace(/^data:image\/png;base64,/, "");
        const dataUrlBuffer = Buffer.from(data, 'base64');
        const result = cv.imdecode(dataUrlBuffer, -1);
        performance.mark("Render - node context-end");
        performance.measure("* Render - node context",
            "Render - node context-init",
            "Render - node context-end");

        return result;
    }

    close() {
        return this.browser.close();
    }
}

module.exports = RenderEngine;
