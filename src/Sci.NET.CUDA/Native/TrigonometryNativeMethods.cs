// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using System.Runtime.InteropServices;
using Sci.NET.Common.Runtime;

namespace Sci.NET.CUDA.Native;

internal static class TrigonometryNativeMethods
{
    static TrigonometryNativeMethods()
    {
        _ = RuntimeDllImportResolver.LoadLibrary(NativeMethods.NativeLibrary, typeof(NativeMethods).Assembly, "CUDA");
    }

    [DllImport(NativeMethods.NativeLibrary, EntryPoint = "sin_fp32", CallingConvention = CallingConvention.Cdecl)]
    public static extern unsafe SdnApiStatusCode SinFp32(float* a, float* b, long n);

    [DllImport(NativeMethods.NativeLibrary, EntryPoint = "sin_fp64", CallingConvention = CallingConvention.Cdecl)]
    public static extern unsafe SdnApiStatusCode SinFp64(double* a, double* b, long n);

    [DllImport(NativeMethods.NativeLibrary, EntryPoint = "cos_fp32", CallingConvention = CallingConvention.Cdecl)]
    public static extern unsafe SdnApiStatusCode CosFp32(float* a, float* b, long n);

    [DllImport(NativeMethods.NativeLibrary, EntryPoint = "cos_fp64", CallingConvention = CallingConvention.Cdecl)]
    public static extern unsafe SdnApiStatusCode CosFp64(double* a, double* b, long n);

    [DllImport(NativeMethods.NativeLibrary, EntryPoint = "tan_fp32", CallingConvention = CallingConvention.Cdecl)]
    public static extern unsafe SdnApiStatusCode TanFp32(float* a, float* b, long n);

    [DllImport(NativeMethods.NativeLibrary, EntryPoint = "tan_fp64", CallingConvention = CallingConvention.Cdecl)]
    public static extern unsafe SdnApiStatusCode TanFp64(double* a, double* b, long n);

    [DllImport(NativeMethods.NativeLibrary, EntryPoint = "sin2_fp32", CallingConvention = CallingConvention.Cdecl)]
    public static extern unsafe SdnApiStatusCode Sin2Fp32(float* a, float* b, long n);

    [DllImport(NativeMethods.NativeLibrary, EntryPoint = "sin2_fp64", CallingConvention = CallingConvention.Cdecl)]
    public static extern unsafe SdnApiStatusCode Sin2Fp64(double* a, double* b, long n);

    [DllImport(NativeMethods.NativeLibrary, EntryPoint = "cos2_fp32", CallingConvention = CallingConvention.Cdecl)]
    public static extern unsafe SdnApiStatusCode Cos2Fp32(float* a, float* b, long n);

    [DllImport(NativeMethods.NativeLibrary, EntryPoint = "cos2_fp64", CallingConvention = CallingConvention.Cdecl)]
    public static extern unsafe SdnApiStatusCode Cos2Fp64(double* a, double* b, long n);

    [DllImport(NativeMethods.NativeLibrary, EntryPoint = "tan2_fp32", CallingConvention = CallingConvention.Cdecl)]
    public static extern unsafe SdnApiStatusCode Tan2Fp32(float* a, float* b, long n);

    [DllImport(NativeMethods.NativeLibrary, EntryPoint = "tan2_fp64", CallingConvention = CallingConvention.Cdecl)]
    public static extern unsafe SdnApiStatusCode Tan2Fp64(double* a, double* b, long n);

    [DllImport(NativeMethods.NativeLibrary, EntryPoint = "sinh_fp32", CallingConvention = CallingConvention.Cdecl)]
    public static extern unsafe SdnApiStatusCode SinhFp32(float* a, float* b, long n);

    [DllImport(NativeMethods.NativeLibrary, EntryPoint = "sinh_fp64", CallingConvention = CallingConvention.Cdecl)]
    public static extern unsafe SdnApiStatusCode SinhFp64(double* a, double* b, long n);

    [DllImport(NativeMethods.NativeLibrary, EntryPoint = "cosh_fp32", CallingConvention = CallingConvention.Cdecl)]
    public static extern unsafe SdnApiStatusCode CoshFp32(float* a, float* b, long n);

    [DllImport(NativeMethods.NativeLibrary, EntryPoint = "cosh_fp64", CallingConvention = CallingConvention.Cdecl)]
    public static extern unsafe SdnApiStatusCode CoshFp64(double* a, double* b, long n);

    [DllImport(NativeMethods.NativeLibrary, EntryPoint = "tanh_fp32", CallingConvention = CallingConvention.Cdecl)]
    public static extern unsafe SdnApiStatusCode TanhFp32(float* a, float* b, long n);

    [DllImport(NativeMethods.NativeLibrary, EntryPoint = "tanh_fp64", CallingConvention = CallingConvention.Cdecl)]
    public static extern unsafe SdnApiStatusCode TanhFp64(double* a, double* b, long n);

    [DllImport(NativeMethods.NativeLibrary, EntryPoint = "sinh2_fp32", CallingConvention = CallingConvention.Cdecl)]
    public static extern unsafe SdnApiStatusCode Sinh2Fp32(float* a, float* b, long n);

    [DllImport(NativeMethods.NativeLibrary, EntryPoint = "sinh2_fp64", CallingConvention = CallingConvention.Cdecl)]
    public static extern unsafe SdnApiStatusCode Sinh2Fp64(double* a, double* b, long n);

    [DllImport(NativeMethods.NativeLibrary, EntryPoint = "cosh2_fp32", CallingConvention = CallingConvention.Cdecl)]
    public static extern unsafe SdnApiStatusCode Cosh2Fp32(float* a, float* b, long n);

    [DllImport(NativeMethods.NativeLibrary, EntryPoint = "cosh2_fp64", CallingConvention = CallingConvention.Cdecl)]
    public static extern unsafe SdnApiStatusCode Cosh2Fp64(double* a, double* b, long n);

    [DllImport(NativeMethods.NativeLibrary, EntryPoint = "tanh2_fp32", CallingConvention = CallingConvention.Cdecl)]
    public static extern unsafe SdnApiStatusCode Tanh2Fp32(float* a, float* b, long n);

    [DllImport(NativeMethods.NativeLibrary, EntryPoint = "tanh2_fp64", CallingConvention = CallingConvention.Cdecl)]
    public static extern unsafe SdnApiStatusCode Tanh2Fp64(double* a, double* b, long n);

    [DllImport(NativeMethods.NativeLibrary, EntryPoint = "asin_fp32", CallingConvention = CallingConvention.Cdecl)]
    public static extern unsafe SdnApiStatusCode AsinFp32(float* a, float* b, long n);

    [DllImport(NativeMethods.NativeLibrary, EntryPoint = "asin_fp64", CallingConvention = CallingConvention.Cdecl)]
    public static extern unsafe SdnApiStatusCode AsinFp64(double* a, double* b, long n);

    [DllImport(NativeMethods.NativeLibrary, EntryPoint = "acos_fp32", CallingConvention = CallingConvention.Cdecl)]
    public static extern unsafe SdnApiStatusCode AcosFp32(float* a, float* b, long n);

    [DllImport(NativeMethods.NativeLibrary, EntryPoint = "acos_fp64", CallingConvention = CallingConvention.Cdecl)]
    public static extern unsafe SdnApiStatusCode AcosFp64(double* a, double* b, long n);

    [DllImport(NativeMethods.NativeLibrary, EntryPoint = "atan_fp32", CallingConvention = CallingConvention.Cdecl)]
    public static extern unsafe SdnApiStatusCode AtanFp32(float* a, float* b, long n);

    [DllImport(NativeMethods.NativeLibrary, EntryPoint = "atan_fp64", CallingConvention = CallingConvention.Cdecl)]
    public static extern unsafe SdnApiStatusCode AtanFp64(double* a, double* b, long n);

    [DllImport(NativeMethods.NativeLibrary, EntryPoint = "asin2_fp32", CallingConvention = CallingConvention.Cdecl)]
    public static extern unsafe SdnApiStatusCode Asin2Fp32(float* a, float* b, long n);

    [DllImport(NativeMethods.NativeLibrary, EntryPoint = "asin2_fp64", CallingConvention = CallingConvention.Cdecl)]
    public static extern unsafe SdnApiStatusCode Asin2Fp64(double* a, double* b, long n);

    [DllImport(NativeMethods.NativeLibrary, EntryPoint = "acos2_fp32", CallingConvention = CallingConvention.Cdecl)]
    public static extern unsafe SdnApiStatusCode Acos2Fp32(float* a, float* b, long n);

    [DllImport(NativeMethods.NativeLibrary, EntryPoint = "acos2_fp64", CallingConvention = CallingConvention.Cdecl)]
    public static extern unsafe SdnApiStatusCode Acos2Fp64(double* a, double* b, long n);

    [DllImport(NativeMethods.NativeLibrary, EntryPoint = "atan2_fp32", CallingConvention = CallingConvention.Cdecl)]
    public static extern unsafe SdnApiStatusCode Atan2Fp32(float* a, float* b, long n);

    [DllImport(NativeMethods.NativeLibrary, EntryPoint = "atan2_fp64", CallingConvention = CallingConvention.Cdecl)]
    public static extern unsafe SdnApiStatusCode Atan2Fp64(double* a, double* b, long n);

    [DllImport(NativeMethods.NativeLibrary, EntryPoint = "asinh_fp32", CallingConvention = CallingConvention.Cdecl)]
    public static extern unsafe SdnApiStatusCode AsinhFp32(float* a, float* b, long n);

    [DllImport(NativeMethods.NativeLibrary, EntryPoint = "asinh_fp64", CallingConvention = CallingConvention.Cdecl)]
    public static extern unsafe SdnApiStatusCode AsinhFp64(double* a, double* b, long n);

    [DllImport(NativeMethods.NativeLibrary, EntryPoint = "acosh_fp32", CallingConvention = CallingConvention.Cdecl)]
    public static extern unsafe SdnApiStatusCode AcoshFp32(float* a, float* b, long n);

    [DllImport(NativeMethods.NativeLibrary, EntryPoint = "acosh_fp64", CallingConvention = CallingConvention.Cdecl)]
    public static extern unsafe SdnApiStatusCode AcoshFp64(double* a, double* b, long n);

    [DllImport(NativeMethods.NativeLibrary, EntryPoint = "atanh_fp32", CallingConvention = CallingConvention.Cdecl)]
    public static extern unsafe SdnApiStatusCode AtanhFp32(float* a, float* b, long n);

    [DllImport(NativeMethods.NativeLibrary, EntryPoint = "atanh_fp64", CallingConvention = CallingConvention.Cdecl)]
    public static extern unsafe SdnApiStatusCode AtanhFp64(double* a, double* b, long n);

    [DllImport(NativeMethods.NativeLibrary, EntryPoint = "asinh2_fp32", CallingConvention = CallingConvention.Cdecl)]
    public static extern unsafe SdnApiStatusCode Asinh2Fp32(float* a, float* b, long n);

    [DllImport(NativeMethods.NativeLibrary, EntryPoint = "asinh2_fp64", CallingConvention = CallingConvention.Cdecl)]
    public static extern unsafe SdnApiStatusCode Asinh2Fp64(double* a, double* b, long n);

    [DllImport(NativeMethods.NativeLibrary, EntryPoint = "acosh2_fp32", CallingConvention = CallingConvention.Cdecl)]
    public static extern unsafe SdnApiStatusCode Acosh2Fp32(float* a, float* b, long n);

    [DllImport(NativeMethods.NativeLibrary, EntryPoint = "acosh2_fp64", CallingConvention = CallingConvention.Cdecl)]
    public static extern unsafe SdnApiStatusCode Acosh2Fp64(double* a, double* b, long n);

    [DllImport(NativeMethods.NativeLibrary, EntryPoint = "atanh2_fp32", CallingConvention = CallingConvention.Cdecl)]
    public static extern unsafe SdnApiStatusCode Atanh2Fp32(float* a, float* b, long n);

    [DllImport(NativeMethods.NativeLibrary, EntryPoint = "atanh2_fp64", CallingConvention = CallingConvention.Cdecl)]
    public static extern unsafe SdnApiStatusCode Atanh2Fp64(double* a, double* b, long n);

    [DllImport(NativeMethods.NativeLibrary, EntryPoint = "csc_fp32", CallingConvention = CallingConvention.Cdecl)]
    public static extern unsafe SdnApiStatusCode CscFp32(float* a, float* b, long n);

    [DllImport(NativeMethods.NativeLibrary, EntryPoint = "csc_fp64", CallingConvention = CallingConvention.Cdecl)]
    public static extern unsafe SdnApiStatusCode CscFp64(double* a, double* b, long n);

    [DllImport(NativeMethods.NativeLibrary, EntryPoint = "sec_fp32", CallingConvention = CallingConvention.Cdecl)]
    public static extern unsafe SdnApiStatusCode SecFp32(float* a, float* b, long n);

    [DllImport(NativeMethods.NativeLibrary, EntryPoint = "sec_fp64", CallingConvention = CallingConvention.Cdecl)]
    public static extern unsafe SdnApiStatusCode SecFp64(double* a, double* b, long n);

    [DllImport(NativeMethods.NativeLibrary, EntryPoint = "cot_fp32", CallingConvention = CallingConvention.Cdecl)]
    public static extern unsafe SdnApiStatusCode CotFp32(float* a, float* b, long n);

    [DllImport(NativeMethods.NativeLibrary, EntryPoint = "cot_fp64", CallingConvention = CallingConvention.Cdecl)]
    public static extern unsafe SdnApiStatusCode CotFp64(double* a, double* b, long n);

    [DllImport(NativeMethods.NativeLibrary, EntryPoint = "csc2_fp32", CallingConvention = CallingConvention.Cdecl)]
    public static extern unsafe SdnApiStatusCode Csc2Fp32(float* a, float* b, long n);

    [DllImport(NativeMethods.NativeLibrary, EntryPoint = "csc2_fp64", CallingConvention = CallingConvention.Cdecl)]
    public static extern unsafe SdnApiStatusCode Csc2Fp64(double* a, double* b, long n);

    [DllImport(NativeMethods.NativeLibrary, EntryPoint = "sec2_fp32", CallingConvention = CallingConvention.Cdecl)]
    public static extern unsafe SdnApiStatusCode Sec2Fp32(float* a, float* b, long n);

    [DllImport(NativeMethods.NativeLibrary, EntryPoint = "sec2_fp64", CallingConvention = CallingConvention.Cdecl)]
    public static extern unsafe SdnApiStatusCode Sec2Fp64(double* a, double* b, long n);

    [DllImport(NativeMethods.NativeLibrary, EntryPoint = "cot2_fp32", CallingConvention = CallingConvention.Cdecl)]
    public static extern unsafe SdnApiStatusCode Cot2Fp32(float* a, float* b, long n);

    [DllImport(NativeMethods.NativeLibrary, EntryPoint = "cot2_fp64", CallingConvention = CallingConvention.Cdecl)]
    public static extern unsafe SdnApiStatusCode Cot2Fp64(double* a, double* b, long n);

    [DllImport(NativeMethods.NativeLibrary, EntryPoint = "csch_fp32", CallingConvention = CallingConvention.Cdecl)]
    public static extern unsafe SdnApiStatusCode CschFp32(float* a, float* b, long n);

    [DllImport(NativeMethods.NativeLibrary, EntryPoint = "csch_fp64", CallingConvention = CallingConvention.Cdecl)]
    public static extern unsafe SdnApiStatusCode CschFp64(double* a, double* b, long n);

    [DllImport(NativeMethods.NativeLibrary, EntryPoint = "sech_fp32", CallingConvention = CallingConvention.Cdecl)]
    public static extern unsafe SdnApiStatusCode SechFp32(float* a, float* b, long n);

    [DllImport(NativeMethods.NativeLibrary, EntryPoint = "sech_fp64", CallingConvention = CallingConvention.Cdecl)]
    public static extern unsafe SdnApiStatusCode SechFp64(double* a, double* b, long n);

    [DllImport(NativeMethods.NativeLibrary, EntryPoint = "coth_fp32", CallingConvention = CallingConvention.Cdecl)]
    public static extern unsafe SdnApiStatusCode CothFp32(float* a, float* b, long n);

    [DllImport(NativeMethods.NativeLibrary, EntryPoint = "coth_fp64", CallingConvention = CallingConvention.Cdecl)]
    public static extern unsafe SdnApiStatusCode CothFp64(double* a, double* b, long n);

    [DllImport(NativeMethods.NativeLibrary, EntryPoint = "csch2_fp32", CallingConvention = CallingConvention.Cdecl)]
    public static extern unsafe SdnApiStatusCode Csch2Fp32(float* a, float* b, long n);

    [DllImport(NativeMethods.NativeLibrary, EntryPoint = "csch2_fp64", CallingConvention = CallingConvention.Cdecl)]
    public static extern unsafe SdnApiStatusCode Csch2Fp64(double* a, double* b, long n);

    [DllImport(NativeMethods.NativeLibrary, EntryPoint = "sech2_fp32", CallingConvention = CallingConvention.Cdecl)]
    public static extern unsafe SdnApiStatusCode Sech2Fp32(float* a, float* b, long n);

    [DllImport(NativeMethods.NativeLibrary, EntryPoint = "sech2_fp64", CallingConvention = CallingConvention.Cdecl)]
    public static extern unsafe SdnApiStatusCode Sech2Fp64(double* a, double* b, long n);

    [DllImport(NativeMethods.NativeLibrary, EntryPoint = "coth2_fp32", CallingConvention = CallingConvention.Cdecl)]
    public static extern unsafe SdnApiStatusCode Coth2Fp32(float* a, float* b, long n);

    [DllImport(NativeMethods.NativeLibrary, EntryPoint = "coth2_fp64", CallingConvention = CallingConvention.Cdecl)]
    public static extern unsafe SdnApiStatusCode Coth2Fp64(double* a, double* b, long n);

    [DllImport(NativeMethods.NativeLibrary, EntryPoint = "acsc_fp32", CallingConvention = CallingConvention.Cdecl)]
    public static extern unsafe SdnApiStatusCode AcscFp32(float* a, float* b, long n);

    [DllImport(NativeMethods.NativeLibrary, EntryPoint = "acsc_fp64", CallingConvention = CallingConvention.Cdecl)]
    public static extern unsafe SdnApiStatusCode AcscFp64(double* a, double* b, long n);

    [DllImport(NativeMethods.NativeLibrary, EntryPoint = "asec_fp32", CallingConvention = CallingConvention.Cdecl)]
    public static extern unsafe SdnApiStatusCode AsecFp32(float* a, float* b, long n);

    [DllImport(NativeMethods.NativeLibrary, EntryPoint = "asec_fp64", CallingConvention = CallingConvention.Cdecl)]
    public static extern unsafe SdnApiStatusCode AsecFp64(double* a, double* b, long n);

    [DllImport(NativeMethods.NativeLibrary, EntryPoint = "acot_fp32", CallingConvention = CallingConvention.Cdecl)]
    public static extern unsafe SdnApiStatusCode AcotFp32(float* a, float* b, long n);

    [DllImport(NativeMethods.NativeLibrary, EntryPoint = "acot_fp64", CallingConvention = CallingConvention.Cdecl)]
    public static extern unsafe SdnApiStatusCode AcotFp64(double* a, double* b, long n);

    [DllImport(NativeMethods.NativeLibrary, EntryPoint = "acsc2_fp32", CallingConvention = CallingConvention.Cdecl)]
    public static extern unsafe SdnApiStatusCode Acsc2Fp32(float* a, float* b, long n);

    [DllImport(NativeMethods.NativeLibrary, EntryPoint = "acsc2_fp64", CallingConvention = CallingConvention.Cdecl)]
    public static extern unsafe SdnApiStatusCode Acsc2Fp64(double* a, double* b, long n);

    [DllImport(NativeMethods.NativeLibrary, EntryPoint = "asec2_fp32", CallingConvention = CallingConvention.Cdecl)]
    public static extern unsafe SdnApiStatusCode Asec2Fp32(float* a, float* b, long n);

    [DllImport(NativeMethods.NativeLibrary, EntryPoint = "asec2_fp64", CallingConvention = CallingConvention.Cdecl)]
    public static extern unsafe SdnApiStatusCode Asec2Fp64(double* a, double* b, long n);

    [DllImport(NativeMethods.NativeLibrary, EntryPoint = "acot2_fp32", CallingConvention = CallingConvention.Cdecl)]
    public static extern unsafe SdnApiStatusCode Acot2Fp32(float* a, float* b, long n);

    [DllImport(NativeMethods.NativeLibrary, EntryPoint = "acot2_fp64", CallingConvention = CallingConvention.Cdecl)]
    public static extern unsafe SdnApiStatusCode Acot2Fp64(double* a, double* b, long n);

    [DllImport(NativeMethods.NativeLibrary, EntryPoint = "acscch_fp32", CallingConvention = CallingConvention.Cdecl)]
    public static extern unsafe SdnApiStatusCode AcscchFp32(float* a, float* b, long n);

    [DllImport(NativeMethods.NativeLibrary, EntryPoint = "acscch_fp64", CallingConvention = CallingConvention.Cdecl)]
    public static extern unsafe SdnApiStatusCode AcscchFp64(double* a, double* b, long n);

    [DllImport(NativeMethods.NativeLibrary, EntryPoint = "asech_fp32", CallingConvention = CallingConvention.Cdecl)]
    public static extern unsafe SdnApiStatusCode AsechFp32(float* a, float* b, long n);

    [DllImport(NativeMethods.NativeLibrary, EntryPoint = "asech_fp64", CallingConvention = CallingConvention.Cdecl)]
    public static extern unsafe SdnApiStatusCode AsechFp64(double* a, double* b, long n);

    [DllImport(NativeMethods.NativeLibrary, EntryPoint = "acoth_fp32", CallingConvention = CallingConvention.Cdecl)]
    public static extern unsafe SdnApiStatusCode AcothFp32(float* a, float* b, long n);

    [DllImport(NativeMethods.NativeLibrary, EntryPoint = "acoth_fp64", CallingConvention = CallingConvention.Cdecl)]
    public static extern unsafe SdnApiStatusCode AcothFp64(double* a, double* b, long n);

    [DllImport(NativeMethods.NativeLibrary, EntryPoint = "acscch2_fp32", CallingConvention = CallingConvention.Cdecl)]
    public static extern unsafe SdnApiStatusCode Acscch2Fp32(float* a, float* b, long n);

    [DllImport(NativeMethods.NativeLibrary, EntryPoint = "acscch2_fp64", CallingConvention = CallingConvention.Cdecl)]
    public static extern unsafe SdnApiStatusCode Acscch2Fp64(double* a, double* b, long n);

    [DllImport(NativeMethods.NativeLibrary, EntryPoint = "asech2_fp32", CallingConvention = CallingConvention.Cdecl)]
    public static extern unsafe SdnApiStatusCode Asech2Fp32(float* a, float* b, long n);

    [DllImport(NativeMethods.NativeLibrary, EntryPoint = "asech2_fp64", CallingConvention = CallingConvention.Cdecl)]
    public static extern unsafe SdnApiStatusCode Asech2Fp64(double* a, double* b, long n);

    [DllImport(NativeMethods.NativeLibrary, EntryPoint = "acoth2_fp32", CallingConvention = CallingConvention.Cdecl)]
    public static extern unsafe SdnApiStatusCode Acoth2Fp32(float* a, float* b, long n);

    [DllImport(NativeMethods.NativeLibrary, EntryPoint = "acoth2_fp64", CallingConvention = CallingConvention.Cdecl)]
    public static extern unsafe SdnApiStatusCode Acoth2Fp64(double* a, double* b, long n);
}