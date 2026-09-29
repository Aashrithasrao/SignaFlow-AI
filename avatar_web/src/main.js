import * as THREE from 'three';

import {
    GLTFLoader
} from 'three/addons/loaders/GLTFLoader.js';

import {
    VRMLoaderPlugin,
    VRMUtils
} from '@pixiv/three-vrm';


// ============================================================
// SCENE
// ============================================================

const scene = new THREE.Scene();

scene.background = new THREE.Color(0x05070c);


// ============================================================
// CAMERA
// ============================================================

const camera = new THREE.PerspectiveCamera(
    28,
    window.innerWidth / window.innerHeight,
    0.1,
    100
);

camera.position.set(
    0,
    1.35,
    3.0
);

camera.lookAt(
    0,
    1.35,
    0
);


// ============================================================
// RENDERER
// ============================================================

const renderer =
    new THREE.WebGLRenderer({
        antialias: true,
        alpha: true
    });

renderer.setPixelRatio(
    Math.min(window.devicePixelRatio, 2)
);

renderer.setSize(
    window.innerWidth,
    window.innerHeight
);

renderer.outputColorSpace =
    THREE.SRGBColorSpace;

renderer.toneMapping =
    THREE.ACESFilmicToneMapping;

renderer.toneMappingExposure = 1.15;

document
    .getElementById("app")
    .appendChild(renderer.domElement);


// ============================================================
// LIGHTING
// ============================================================

const keyLight =
    new THREE.DirectionalLight(
        0xffffff,
        3.0
    );

keyLight.position.set(
    2,
    4,
    4
);

scene.add(keyLight);


const fillLight =
    new THREE.DirectionalLight(
        0x9fb7ff,
        1.5
    );

fillLight.position.set(
    -3,
    2,
    2
);

scene.add(fillLight);


const rimLight =
    new THREE.DirectionalLight(
        0xffffff,
        1.5
    );

rimLight.position.set(
    0,
    3,
    -4
);

scene.add(rimLight);


scene.add(
    new THREE.HemisphereLight(
        0xffffff,
        0x10131c,
        1.3
    )
);


// ============================================================
// FLOOR
// ============================================================

const floor =
    new THREE.Mesh(
        new THREE.CircleGeometry(
            3,
            64
        ),
        new THREE.MeshStandardMaterial({
            color: 0x090c13,
            roughness: 0.92,
            metalness: 0.05
        })
    );

floor.rotation.x =
    -Math.PI / 2;

scene.add(floor);


// ============================================================
// VRM
// ============================================================

let vrm = null;
let humanoid = null;

const status =
    document.getElementById("status");


// ============================================================
// LOAD
// ============================================================

async function loadAvatar() {

    const loader =
        new GLTFLoader();

    loader.crossOrigin = "anonymous";

    loader.register(
        parser =>
            new VRMLoaderPlugin(parser)
    );

    try {

        const gltf =
            await loader.loadAsync(
                "/models/avatar.vrm.glb"
            );

        vrm =
            gltf.userData.vrm;

        if (!vrm) {
            throw new Error(
                "VRM model was not detected."
            );
        }

        VRMUtils.removeUnnecessaryVertices(
            vrm.scene
        );

        VRMUtils.removeUnnecessaryJoints(
            vrm.scene
        );

        vrm.scene.traverse(
            object => {
                object.frustumCulled = false;
            }
        );

        scene.add(vrm.scene);

        humanoid =
            vrm.humanoid;

        console.log(
            "SignaFlow VRM:",
            vrm
        );

        printBones();

        status.textContent =
            "Avatar ready";

    } catch (error) {

        console.error(error);

        status.textContent =
            "Avatar loading failed";
    }
}


// ============================================================
// BONE
// ============================================================

function bone(name) {

    if (!humanoid) {
        return null;
    }

    return humanoid.getNormalizedBoneNode(
        name
    );
}


// ============================================================
// BONE REPORT
// ============================================================

function printBones() {

    const important = [

        "head",

        "leftUpperArm",
        "leftLowerArm",
        "leftHand",

        "rightUpperArm",
        "rightLowerArm",
        "rightHand",

        "rightThumbMetacarpal",
        "rightThumbProximal",
        "rightThumbDistal",

        "rightIndexProximal",
        "rightIndexIntermediate",
        "rightIndexDistal",

        "rightMiddleProximal",
        "rightMiddleIntermediate",
        "rightMiddleDistal",

        "rightRingProximal",
        "rightRingIntermediate",
        "rightRingDistal",

        "rightLittleProximal",
        "rightLittleIntermediate",
        "rightLittleDistal"
    ];

    console.log(
        "========== SIGNaflow BONE TEST =========="
    );

    for (const name of important) {

        console.log(
            name,
            bone(name)
                ? "✓"
                : "✗"
        );
    }
}


// ============================================================
// FINGER DATA
// ============================================================

const fingerGroups = {

    thumb: [
        "ThumbMetacarpal",
        "ThumbProximal",
        "ThumbDistal"
    ],

    index: [
        "IndexProximal",
        "IndexIntermediate",
        "IndexDistal"
    ],

    middle: [
        "MiddleProximal",
        "MiddleIntermediate",
        "MiddleDistal"
    ],

    ring: [
        "RingProximal",
        "RingIntermediate",
        "RingDistal"
    ],

    little: [
        "LittleProximal",
        "LittleIntermediate",
        "LittleDistal"
    ]
};


// ============================================================
// RESET FINGERS
// ============================================================

function resetHand(side) {

    for (
        const joints of Object.values(
            fingerGroups
        )
    ) {

        for (const joint of joints) {

            const b =
                bone(
                    side + joint
                );

            if (!b) continue;

            b.rotation.set(
                0,
                0,
                0
            );
        }
    }
}


// ============================================================
// FINGER CURL
// ============================================================

function curl(
    side,
    finger,
    amount
) {

    const joints =
        fingerGroups[finger];

    if (!joints) return;

    joints.forEach(
        (joint, i) => {

            const b =
                bone(
                    side + joint
                );

            if (!b) return;

            /*
             * We deliberately use the local X axis.
             * VRM normalized finger bones expose
             * their local articulation through this axis
             * on this avatar.
             */

            b.rotation.x =
                amount *
                (i === 0 ? 0.85 : 1.10);
        }
    );
}


// ============================================================
// W HANDSHAPE
// ============================================================

function makeWHand() {

    if (!humanoid) return;

    const side = "right";

    resetHand(side);

    /*
     * W:
     *
     * index  : extended
     * middle : extended
     * ring   : extended
     * little : folded
     * thumb  : across / toward little
     */

    curl(
        side,
        "little",
        0.95
    );

    const thumbProx =
        bone(
            "rightThumbProximal"
        );

    const thumbDist =
        bone(
            "rightThumbDistal"
        );

    if (thumbProx) {

        thumbProx.rotation.x =
            0.35;

        thumbProx.rotation.y =
            -0.45;
    }

    if (thumbDist) {

        thumbDist.rotation.x =
            0.25;
    }

    showWord("W");
}


// ============================================================
// OPEN
// ============================================================

function openBothHands() {

    if (!humanoid) return;

    resetHand("left");
    resetHand("right");

    showWord("OPEN");
}


// ============================================================
// FIST
// ============================================================

function fistBothHands() {

    if (!humanoid) return;

    resetHand("left");
    resetHand("right");

    for (const side of [
        "left",
        "right"
    ]) {

        curl(side, "index", 1);
        curl(side, "middle", 1);
        curl(side, "ring", 1);
        curl(side, "little", 1);
    }

    showWord("FIST");
}


// ============================================================
// POINT
// ============================================================

function pointBothHands() {

    if (!humanoid) return;

    resetHand("left");
    resetHand("right");

    for (const side of [
        "left",
        "right"
    ]) {

        curl(side, "middle", 1);
        curl(side, "ring", 1);
        curl(side, "little", 1);
    }

    showWord("POINT");
}


// ============================================================
// MOVE RIGHT ARM
// ============================================================

function prepareRightHandNearFace() {

    const upper =
        bone("rightUpperArm");

    const lower =
        bone("rightLowerArm");

    const hand =
        bone("rightHand");

    if (!upper || !lower || !hand) {
        return;
    }

    /*
     * Bring the dominant arm upward
     * toward the face.
     *
     * These values are intentionally
     * moderate so we can calibrate
     * against the current avatar.
     */

    upper.rotation.z =
        -1.15;

    upper.rotation.y =
        0.20;

    lower.rotation.z =
        -0.55;

    lower.rotation.y =
        -0.20;

    hand.rotation.z =
        0.15;
}


// ============================================================
// RETURN RIGHT ARM
// ============================================================

function resetRightArm() {

    const upper =
        bone("rightUpperArm");

    const lower =
        bone("rightLowerArm");

    const hand =
        bone("rightHand");

    if (upper) {
        upper.rotation.set(
            0,
            0,
            0
        );
    }

    if (lower) {
        lower.rotation.set(
            0,
            0,
            0
        );
    }

    if (hand) {
        hand.rotation.set(
            0,
            0,
            0
        );
    }
}


// ============================================================
// WATER ANIMATION
// ============================================================

let waterRunning = false;


function avatarWater() {

    if (!humanoid || waterRunning) {
        return;
    }

    waterRunning = true;

    showWord("WATER");

    /*
     * Start from a clean pose.
     */

    resetRightArm();

    resetHand("right");

    /*
     * Create W handshape.
     */

    makeWHand();

    /*
     * Move the hand toward the chin.
     */

    prepareRightHandNearFace();

    const index =
        bone(
            "rightIndexDistal"
        );

    const indexMiddle =
        bone(
            "rightIndexIntermediate"
        );

    /*
     * Keep index extended.
     */

    if (index) {
        index.rotation.x = 0;
    }

    if (indexMiddle) {
        indexMiddle.rotation.x = 0;
    }

    /*
     * Two taps.
     */

    const hand =
        bone("rightHand");

    const start =
        performance.now();

    const duration = 1900;

    function animate(now) {

        const elapsed =
            now - start;

        const progress =
            Math.min(
                1,
                elapsed / duration
            );

        /*
         * Two short forward/back
         * movements.
         */

        const tap =
            Math.sin(
                progress *
                Math.PI *
                4
            );

        if (hand) {

            hand.position.z =
                tap * 0.025;
        }

        if (progress < 1) {

            requestAnimationFrame(
                animate
            );

        } else {

            if (hand) {
                hand.position.set(
                    0,
                    0,
                    0
                );
            }

            resetRightArm();

            resetHand("right");

            showWord("");

            waterRunning = false;
        }
    }

    requestAnimationFrame(
        animate
    );
}


// ============================================================
// RESET
// ============================================================

function avatarReset() {

    if (!humanoid) return;

    humanoid.resetNormalizedPose();

    showWord("");
}


// ============================================================
// LABEL
// ============================================================

function showWord(text) {

    const element =
        document.getElementById(
            "word"
        );

    element.textContent =
        text;

    if (text) {
        element.classList.add("show");
    } else {
        element.classList.remove("show");
    }
}


// ============================================================
// RENDER
// ============================================================

const clock =
    new THREE.Clock();

function animate() {

    requestAnimationFrame(
        animate
    );

    const delta =
        clock.getDelta();

    if (vrm) {
        vrm.update(delta);
    }

    renderer.render(
        scene,
        camera
    );
}


// ============================================================
// RESIZE
// ============================================================

window.addEventListener(
    "resize",
    () => {

        camera.aspect =
            window.innerWidth /
            window.innerHeight;

        camera.updateProjectionMatrix();

        renderer.setSize(
            window.innerWidth,
            window.innerHeight
        );
    }
);


// ============================================================
// GLOBAL BUTTONS
// ============================================================

window.avatarOpen =
    openBothHands;

window.avatarFist =
    fistBothHands;

window.avatarPoint =
    pointBothHands;

window.avatarWater =
    avatarWater;

window.avatarReset =
    avatarReset;


// ============================================================
// START
// ============================================================

loadAvatar();

animate();
