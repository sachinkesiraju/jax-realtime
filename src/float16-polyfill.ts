// Polyfill Float16Array for browsers that do not yet support it (e.g. Chrome
// < 135, where it first shipped). jax-realtime and jax-js use Float16Array for
// half-precision weight buffers, so a missing global breaks model loading.
//
// This is a minimal, jax-js-compatible implementation: it stores raw fp16 bits
// in a Uint16Array, converts to/from Number on access, and masquerades as a
// real TypedArray for ArrayBuffer.isView / instanceof checks. It is only
// installed when the native Float16Array is absent.

type Float16ArrayLike = {
  readonly buffer: ArrayBuffer;
  readonly byteOffset: number;
  readonly byteLength: number;
  readonly length: number;
  [Symbol.toStringTag]: "Float16Array";
};

const floatView = new Float32Array(1);
const int32View = new Int32Array(floatView.buffer);

function toHalfBits(val: number): number {
  floatView[0] = val;
  const x = int32View[0];

  let bits = (x >> 16) & 0x8000; // sign
  const m = (x >> 12) & 0x07ff; // mantissa with one extra rounding bit
  const e = (x >> 23) & 0xff; // exponent

  if (e < 103) {
    return bits; // zero / underflow
  }
  if (e > 142) {
    bits |= 0x7c00; // Inf
    // NaN if original was NaN, otherwise keep mantissa zero
    bits |= e === 255 && (x & 0x007fffff) !== 0 ? 0x200 : 0;
    return bits;
  }
  if (e < 113) {
    // denormalized
    bits |= (m | 0x0800) >> (114 - e);
    return bits;
  }
  bits |= ((e - 112) << 10) | (m >> 1);
  // round to nearest even
  bits += m & 1;
  return bits & 0xffff;
}

function fromHalfBits(h: number): number {
  const sign = (h >> 15) & 1;
  const exponent = (h >> 10) & 0x1f;
  const fraction = h & 0x3ff;

  if (exponent === 0) {
    if (fraction === 0) return sign ? -0 : 0;
    // subnormal: fraction / 2^10 * 2^-14
    const v = 6.103515625e-5 * (fraction / 1024);
    return sign ? -v : v;
  }
  if (exponent === 0x1f) {
    return fraction ? NaN : sign ? -Infinity : Infinity;
  }
  const v = Math.pow(2, exponent - 15) * (1 + fraction / 1024);
  return sign ? -v : v;
}

function isFloat16Instance(obj: unknown): obj is Float16ArrayLike {
  return (
    obj !== null &&
    typeof obj === "object" &&
    (obj as { [Symbol.toStringTag]?: string })[Symbol.toStringTag] ===
      "Float16Array"
  );
}

function isArrayBufferView(obj: unknown): obj is ArrayBufferView {
  return obj !== null && typeof obj === "object" && ArrayBuffer.isView(obj);
}

type Float16Arg =
  | number
  | ArrayLike<number>
  | ArrayBuffer
  | ArrayBufferView
  | Iterable<number>
  | Float16ArrayLike;

class Float16ArrayImpl {
  static BYTES_PER_ELEMENT = 2;

  static from<T>(
    source: Iterable<T> | ArrayLike<T>,
    mapFn?: (v: T, k: number) => number,
    thisArg?: unknown,
  ): unknown {
    const arr = Array.from(
      source,
      mapFn as (v: T, k: number) => number,
      thisArg,
    );
    return new Float16ArrayImpl(arr);
  }

  static of(...items: number[]): unknown {
    return new Float16ArrayImpl(items);
  }

  [Symbol.toStringTag] = "Float16Array";

  constructor(...args: unknown[]) {
    let source: Uint16Array;

    if (args.length === 0) {
      source = new Uint16Array(0);
    } else if (args.length === 1) {
      const arg = args[0];
      if (typeof arg === "number") {
        source = new Uint16Array(arg);
      } else if (isFloat16Instance(arg)) {
        source = new Uint16Array(arg.buffer, arg.byteOffset, arg.length);
      } else if (arg instanceof ArrayBuffer) {
        source = new Uint16Array(arg);
      } else if (ArrayBuffer.isView(arg)) {
        const view = arg as unknown as { length: number; [i: number]: number };
        source = new Uint16Array(view.length);
        for (let i = 0; i < view.length; i++) {
          source[i] = toHalfBits(view[i]);
        }
      } else if (Array.isArray(arg) || typeof (arg as Iterable<unknown>)[Symbol.iterator] === "function") {
        const values = Array.isArray(arg) ? (arg as number[]) : Array.from(arg as Iterable<number>);
        source = new Uint16Array(values.length);
        for (let i = 0; i < values.length; i++) {
          source[i] = toHalfBits(values[i] as number);
        }
      } else {
        source = new Uint16Array(0);
      }
    } else {
      const [buf, byteOffsetArg, lengthArg] = args;
      const byteOffset = (byteOffsetArg as number | undefined) ?? 0;
      const length = lengthArg as number | undefined;
      const isAb = buf instanceof ArrayBuffer;
      const ab = isAb ? (buf as ArrayBuffer) : ((buf as ArrayBufferView).buffer as ArrayBuffer);
      const view = buf as ArrayBufferView;
      const offset = isAb ? byteOffset : byteOffset + (view.byteOffset || 0);
      const len =
        length !== undefined
          ? length
          : (view.byteLength - offset) / 2;
      source = new Uint16Array(ab, offset, len);
    }

    const proxy = new Proxy(source, {
      get(target, prop) {
        if (prop === Symbol.toStringTag) return "Float16Array";
        if (prop === Symbol.iterator) {
          return function* () {
            for (let i = 0; i < target.length; i++) {
              yield fromHalfBits(target[i]!);
            }
          };
        }
        if (prop === "buffer") return target.buffer;
        if (prop === "byteOffset") return target.byteOffset;
        if (prop === "byteLength") return target.byteLength;
        if (prop === "length") return target.length;
        if (prop === "constructor") return Float16ArrayImpl;
        if (prop === "BYTES_PER_ELEMENT") return 2;
        if (
          prop === Symbol.toPrimitive ||
          prop === "toString" ||
          prop === "valueOf"
        ) {
          return () => "[object Float16Array]";
        }
        if (typeof prop === "string" && /^\d+$/.test(prop)) {
          const idx = Number(prop);
          if (idx >= 0 && idx < target.length) {
            return fromHalfBits(target[idx]!);
          }
        }
        const value = (target as unknown as Record<string | symbol, unknown>)[
          prop
        ];
        if (typeof value === "function") {
          return value.bind(target);
        }
        return value;
      },
      set(target, prop, value) {
        if (typeof prop === "string" && /^\d+$/.test(prop)) {
          const idx = Number(prop);
          if (idx >= 0 && idx < target.length) {
            target[idx] = toHalfBits(Number(value));
            return true;
          }
        }
        (target as unknown as Record<string | symbol, unknown>)[prop] = value;
        return true;
      },
      has(target, prop) {
        return prop in target;
      },
      ownKeys(target) {
        const keys: string[] = [];
        for (let i = 0; i < target.length; i++) keys.push(String(i));
        return keys;
      },
      getPrototypeOf() {
        return Float16ArrayImpl.prototype;
      },
    });

    Object.setPrototypeOf(proxy, Float16ArrayImpl.prototype);
    return proxy as unknown as Float16Array;
  }
}

// Ensure Float16Array appears to extend Uint16Array for instanceof checks.
Object.setPrototypeOf(Float16ArrayImpl.prototype, Uint16Array.prototype);
Object.setPrototypeOf(Float16ArrayImpl, Uint16Array);

export function installFloat16Polyfill(): void {
  if (typeof (globalThis as { Float16Array?: unknown }).Float16Array !== "undefined") {
    return;
  }

  (globalThis as { Float16Array: unknown }).Float16Array =
    Float16ArrayImpl as unknown as Float16Array;

  // jax-js checks ArrayBuffer.isView before dispatching to arrayFromData.
  const originalIsView = ArrayBuffer.isView;
  Object.defineProperty(ArrayBuffer, "isView", {
    value: (v: unknown) =>
      isFloat16Instance(v) || originalIsView(v),
    writable: true,
    configurable: true,
  });
}

installFloat16Polyfill();
