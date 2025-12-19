import * as THREE from 'three';

// Improved Simplex Noise implementation for 3D anatomy generation
// Based on standard noise algorithms but optimized for typed arrays

class SimplexNoise {
    private perm: Uint8Array;
    private grad3: Float32Array;

    constructor() {
        this.perm = new Uint8Array(512);
        this.grad3 = new Float32Array([
            1, 1, 0, -1, 1, 0, 1, -1, 0, -1, -1, 0,
            1, 0, 1, -1, 0, 1, 1, 0, -1, -1, 0, -1,
            0, 1, 1, 0, -1, 1, 0, 1, -1, 0, -1, -1
        ]);

        // Initialize permutation table
        const p = new Uint8Array(256);
        for (let i = 0; i < 256; i++) p[i] = i;

        // Shuffle
        for (let i = 255; i > 0; i--) {
            const r = Math.floor(Math.random() * (i + 1));
            const t = p[i]; p[i] = p[r]; p[r] = t;
        }

        // Duplicate for wrapping
        for (let i = 0; i < 512; i++) this.perm[i] = p[i & 255];
    }

    dot(g: Float32Array, x: number, y: number, z: number, idx: number): number {
        return g[idx] * x + g[idx + 1] * y + g[idx + 2] * z;
    }

    noise3D(xin: number, yin: number, zin: number): number {
        let n0, n1, n2, n3; // Noise contributions from the four corners

        // Skew the input space to determine which simplex cell we're in
        const F3 = 1.0 / 3.0;
        const s = (xin + yin + zin) * F3;
        const i = Math.floor(xin + s);
        const j = Math.floor(yin + s);
        const k = Math.floor(zin + s);

        const G3 = 1.0 / 6.0;
        const t = (i + j + k) * G3;
        const X0 = i - t;
        const Y0 = j - t;
        const Z0 = k - t;

        const x0 = xin - X0;
        const y0 = yin - Y0;
        const z0 = zin - Z0;

        // For the 3D case, the simplex shape is a slightly irregular tetrahedron.
        // Determine which simplex we are in.
        let i1, j1, k1; // Offsets for second corner of simplex in (i,j,k) coords
        let i2, j2, k2; // Offsets for third corner of simplex in (i,j,k) coords

        if (x0 >= y0) {
            if (y0 >= z0) { i1 = 1; j1 = 0; k1 = 0; i2 = 1; j2 = 1; k2 = 0; } // X Y Z order
            else if (x0 >= z0) { i1 = 1; j1 = 0; k1 = 0; i2 = 1; j2 = 0; k2 = 1; } // X Z Y order
            else { i1 = 0; j1 = 0; k1 = 1; i2 = 1; j2 = 0; k2 = 1; } // Z X Y order
        } else { // x0<y0
            if (y0 < z0) { i1 = 0; j1 = 0; k1 = 1; i2 = 0; j2 = 1; k2 = 1; } // Z Y X order
            else if (x0 < z0) { i1 = 0; j1 = 1; k1 = 0; i2 = 0; j2 = 1; k2 = 1; } // Y Z X order
            else { i1 = 0; j1 = 1; k1 = 0; i2 = 1; j2 = 1; k2 = 0; } // Y X Z order
        }

        // A step of (1,0,0) in (i,j,k) means a step of (1-c,-c,-c) in (x,y,z),
        // a step of (0,1,0) in (i,j,k) means a step of (-c,1-c,-c) in (x,y,z), and
        // a step of (0,0,1) in (i,j,k) means a step of (-c,-c,1-c) in (x,y,z), where
        // c = 1/6.

        const x1 = x0 - i1 + G3; // Offsets for second corner in (x,y,z) coords
        const y1 = y0 - j1 + G3;
        const z1 = z0 - k1 + G3;
        const x2 = x0 - i2 + 2.0 * G3; // Offsets for third corner in (x,y,z) coords
        const y2 = y0 - j2 + 2.0 * G3;
        const z2 = z0 - k2 + 2.0 * G3;
        const x3 = x0 - 1.0 + 3.0 * G3; // Offsets for last corner in (x,y,z) coords
        const y3 = y0 - 1.0 + 3.0 * G3;
        const z3 = z0 - 1.0 + 3.0 * G3;

        // Work out the hashed gradient indices of the four simplex corners
        const ii = i & 255;
        const jj = j & 255;
        const kk = k & 255;

        const gi0 = this.perm[ii + this.perm[jj + this.perm[kk]]] % 12;
        const gi1 = this.perm[ii + i1 + this.perm[jj + j1 + this.perm[kk + k1]]] % 12;
        const gi2 = this.perm[ii + i2 + this.perm[jj + j2 + this.perm[kk + k2]]] % 12;
        const gi3 = this.perm[ii + 1 + this.perm[jj + 1 + this.perm[kk + 1]]] % 12;

        // Calculate the contribution from the four corners
        let t0 = 0.6 - x0 * x0 - y0 * y0 - z0 * z0;
        if (t0 < 0) n0 = 0.0;
        else {
            t0 *= t0;
            n0 = t0 * t0 * this.dot(this.grad3, x0, y0, z0, gi0 * 3);
        }

        let t1 = 0.6 - x1 * x1 - y1 * y1 - z1 * z1;
        if (t1 < 0) n1 = 0.0;
        else {
            t1 *= t1;
            n1 = t1 * t1 * this.dot(this.grad3, x1, y1, z1, gi1 * 3);
        }

        let t2 = 0.6 - x2 * x2 - y2 * y2 - z2 * z2;
        if (t2 < 0) n2 = 0.0;
        else {
            t2 *= t2;
            n2 = t2 * t2 * this.dot(this.grad3, x2, y2, z2, gi2 * 3);
        }

        let t3 = 0.6 - x3 * x3 - y3 * y3 - z3 * z3;
        if (t3 < 0) n3 = 0.0;
        else {
            t3 *= t3;
            n3 = t3 * t3 * this.dot(this.grad3, x3, y3, z3, gi3 * 3);
        }

        // Add contributions from each corner to get the final noise value.
        // The result is scaled to stay just inside [-1,1]
        return 32.0 * (n0 + n1 + n2 + n3);
    }
}

const noise = new SimplexNoise();

export function generateBrainVolume(size: number = 128): Float32Array {
    const data = new Float32Array(size * size * size);
    const center = size / 2;

    for (let z = 0; z < size; z++) {
        for (let y = 0; y < size; y++) {
            for (let x = 0; x < size; x++) {
                // Normalized coordinates [-1, 1]
                const nx = (x - center) / center;
                const ny = (y - center) / center;
                const nz = (z - center) / center;

                const r = Math.sqrt(nx * nx + ny * ny + nz * nz);
                const index = z * size * size + y * size + x;

                // 1. Base Skull Shape (Ellipsoid)
                let density = 0;

                // Skull boundary
                // Elongate slightly in Z (front-to-back)
                const shape = Math.sqrt(nx * nx * 0.8 + ny * ny + nz * nz * 0.9);

                if (shape < 0.95 && shape > 0.85) {
                    density = 0.8 + Math.random() * 0.1; // Bone
                } else if (shape <= 0.85) {
                    // Brain Mass

                    // 2. Gyri/Sulci Construction (Detailed Noise)
                    const coarse = noise.noise3D(nx * 2.5, ny * 2.5, nz * 2.5);
                    const fine = noise.noise3D(nx * 8.0, ny * 8.0, nz * 8.0);
                    const micro = noise.noise3D(nx * 20.0, ny * 20.0, nz * 20.0);

                    // Layered noise for tissue structure
                    const structure = coarse * 0.6 + fine * 0.3 + micro * 0.1;

                    // 3. Ventricles (Butterfly shape void)
                    const vx = Math.abs(nx);
                    const vy = ny + 0.1;
                    const vz = nz;
                    const ventricleDist = Math.sqrt(vx * vx / 0.05 + vy * vy / 0.2 + vz * vz / 0.1);

                    let tissueDensity = 0.45; // Gray Matter Base

                    // White Matter cores (deeper inside)
                    if (shape < 0.6) {
                        tissueDensity += (structure * 0.2) + 0.2; // Brighter white matter
                    } else {
                        tissueDensity += (structure * 0.1); // Cortex variation
                    }

                    // Ventricle Carving (CSF is dark/fluid)
                    if (ventricleDist < 0.6) {
                        tissueDensity = 0.1 + Math.random() * 0.05; // Fluid
                    }

                    // Longitudinal Fissure (Split L/R hemispheres)
                    if (Math.abs(nx) < 0.02 && shape < 0.8) {
                        tissueDensity = 0.0;
                    }

                    density = tissueDensity;
                }

                // 4. Soft Clamp
                density = Math.max(0, Math.min(1, density));

                data[index] = density;
            }
        }
    }

    return data;
}
