/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Global Instruction Selector for the X86 target                             *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

#ifdef GET_GLOBALISEL_PREDICATE_BITSET
const unsigned MAX_SUBTARGET_PREDICATES = 113;
using PredicateBitset = llvm::PredicateBitsetImpl<MAX_SUBTARGET_PREDICATES>;
#endif // ifdef GET_GLOBALISEL_PREDICATE_BITSET

#ifdef GET_GLOBALISEL_TEMPORARIES_DECL
  mutable MatcherState State;
  typedef ComplexRendererFns(X86InstructionSelector::*ComplexMatcherMemFn)(MachineOperand &) const;
  typedef void(X86InstructionSelector::*CustomRendererFn)(MachineInstrBuilder &, const MachineInstr&) const;
  const ISelInfoTy<PredicateBitset, ComplexMatcherMemFn, CustomRendererFn> ISelInfo;
  static X86InstructionSelector::ComplexMatcherMemFn ComplexPredicateFns[];
  static X86InstructionSelector::CustomRendererFn CustomRenderers[];
  bool testImmPredicate_I64(unsigned PredicateID, int64_t Imm) const override;
  bool testImmPredicate_APInt(unsigned PredicateID, const APInt &Imm) const override;
  bool testImmPredicate_APFloat(unsigned PredicateID, const APFloat &Imm) const override;
  const int64_t *getMatchTable() const override;
  bool testMIPredicate_MI(unsigned PredicateID, const MachineInstr &MI) const override;
#endif // ifdef GET_GLOBALISEL_TEMPORARIES_DECL

#ifdef GET_GLOBALISEL_TEMPORARIES_INIT
, State(0),
ISelInfo(TypeObjects, NumTypeObjects, FeatureBitsets, ComplexPredicateFns, CustomRenderers)
#endif // ifdef GET_GLOBALISEL_TEMPORARIES_INIT

#ifdef GET_GLOBALISEL_IMPL
// Bits for subtarget features that participate in instruction matching.
enum SubtargetFeatureBits : uint8_t {
  Feature_TruePredicateBit = 48,
  Feature_HasCMovBit = 20,
  Feature_NoCMovBit = 100,
  Feature_HasMMXBit = 81,
  Feature_Has3DNowBit = 83,
  Feature_HasSSE1Bit = 34,
  Feature_UseSSE1Bit = 42,
  Feature_HasSSE2Bit = 35,
  Feature_UseSSE2Bit = 43,
  Feature_HasSSE3Bit = 26,
  Feature_UseSSE3Bit = 51,
  Feature_HasSSSE3Bit = 82,
  Feature_UseSSSE3Bit = 52,
  Feature_UseSSE41Bit = 49,
  Feature_HasSSE42Bit = 56,
  Feature_UseSSE42Bit = 55,
  Feature_HasSSE4ABit = 65,
  Feature_NoAVXBit = 61,
  Feature_HasAVXBit = 44,
  Feature_HasAVX2Bit = 38,
  Feature_HasAVX1OnlyBit = 36,
  Feature_HasAVX512Bit = 69,
  Feature_UseAVXBit = 40,
  Feature_UseAVX2Bit = 67,
  Feature_NoAVX512Bit = 31,
  Feature_HasCDIBit = 73,
  Feature_HasVPOPCNTDQBit = 77,
  Feature_HasERIBit = 76,
  Feature_HasDQIBit = 71,
  Feature_NoDQIBit = 53,
  Feature_HasBWIBit = 72,
  Feature_NoBWIBit = 50,
  Feature_HasVLXBit = 70,
  Feature_NoVLXBit = 30,
  Feature_NoVLX_Or_NoBWIBit = 47,
  Feature_NoVLX_Or_NoDQIBit = 104,
  Feature_HasVNNIBit = 79,
  Feature_HasBITALGBit = 80,
  Feature_HasPOPCNTBit = 54,
  Feature_HasAESBit = 58,
  Feature_HasVAESBit = 60,
  Feature_NoVLX_Or_NoVAESBit = 59,
  Feature_HasFXSRBit = 27,
  Feature_HasXSAVEBit = 89,
  Feature_HasXSAVEOPTBit = 90,
  Feature_HasXSAVECBit = 91,
  Feature_HasXSAVESBit = 92,
  Feature_HasPCLMULBit = 62,
  Feature_NoVLX_Or_NoVPCLMULQDQBit = 63,
  Feature_HasVPCLMULQDQBit = 64,
  Feature_HasGFNIBit = 68,
  Feature_HasFMABit = 28,
  Feature_HasFMA4Bit = 32,
  Feature_NoFMA4Bit = 29,
  Feature_HasXOPBit = 33,
  Feature_HasTBMBit = 9,
  Feature_NoTBMBit = 111,
  Feature_HasLWPBit = 10,
  Feature_HasMOVBEBit = 3,
  Feature_HasRDRANDBit = 4,
  Feature_HasF16CBit = 66,
  Feature_HasFSGSBaseBit = 93,
  Feature_HasLZCNTBit = 6,
  Feature_HasBMIBit = 7,
  Feature_HasBMI2Bit = 8,
  Feature_NoBMI2Bit = 112,
  Feature_HasVBMIBit = 74,
  Feature_HasVBMI2Bit = 78,
  Feature_HasIFMABit = 75,
  Feature_HasRTMBit = 87,
  Feature_HasSHABit = 57,
  Feature_HasRDSEEDBit = 5,
  Feature_HasSSEPrefetchBit = 45,
  Feature_NoSSEPrefetchBit = 84,
  Feature_HasPrefetchWBit = 85,
  Feature_HasPREFETCHWT1Bit = 86,
  Feature_HasLAHFSAHFBit = 2,
  Feature_HasMWAITXBit = 11,
  Feature_HasCLZEROBit = 15,
  Feature_HasCLDEMOTEBit = 18,
  Feature_HasMOVDIRIBit = 13,
  Feature_HasMOVDIR64BBit = 14,
  Feature_HasPTWRITEBit = 96,
  Feature_FPStackf32Bit = 24,
  Feature_FPStackf64Bit = 25,
  Feature_HasCLFLUSHOPTBit = 16,
  Feature_HasCLWBBit = 17,
  Feature_HasWBNOINVDBit = 88,
  Feature_HasRDPIDBit = 95,
  Feature_HasWAITPKGBit = 12,
  Feature_HasINVPCIDBit = 94,
  Feature_HasCmpxchg16bBit = 101,
  Feature_Not64BitModeBit = 0,
  Feature_In64BitModeBit = 1,
  Feature_IsLP64Bit = 98,
  Feature_NotLP64Bit = 97,
  Feature_NotWin64WithoutFPBit = 99,
  Feature_IsPS4Bit = 106,
  Feature_NotPS4Bit = 105,
  Feature_KernelCodeBit = 107,
  Feature_NearDataBit = 109,
  Feature_IsNotPICBit = 108,
  Feature_OptForSizeBit = 39,
  Feature_OptForMinSizeBit = 37,
  Feature_OptForSpeedBit = 103,
  Feature_UseIncDecBit = 19,
  Feature_NoSSE41_Or_OptForSizeBit = 41,
  Feature_CallImmAddrBit = 110,
  Feature_FavorMemIndirectCallBit = 21,
  Feature_HasFastSHLDRotateBit = 102,
  Feature_HasMFenceBit = 46,
  Feature_UseRetpolineIndirectCallsBit = 23,
  Feature_NotUseRetpolineIndirectCallsBit = 22,
};

PredicateBitset X86InstructionSelector::
computeAvailableModuleFeatures(const X86Subtarget *Subtarget) const {
  PredicateBitset Features;
  if (true)
    Features[Feature_TruePredicateBit] = 1;
  if (Subtarget->hasCMov())
    Features[Feature_HasCMovBit] = 1;
  if (!Subtarget->hasCMov())
    Features[Feature_NoCMovBit] = 1;
  if (Subtarget->hasMMX())
    Features[Feature_HasMMXBit] = 1;
  if (Subtarget->has3DNow())
    Features[Feature_Has3DNowBit] = 1;
  if (Subtarget->hasSSE1())
    Features[Feature_HasSSE1Bit] = 1;
  if (Subtarget->hasSSE1() && !Subtarget->hasAVX())
    Features[Feature_UseSSE1Bit] = 1;
  if (Subtarget->hasSSE2())
    Features[Feature_HasSSE2Bit] = 1;
  if (Subtarget->hasSSE2() && !Subtarget->hasAVX())
    Features[Feature_UseSSE2Bit] = 1;
  if (Subtarget->hasSSE3())
    Features[Feature_HasSSE3Bit] = 1;
  if (Subtarget->hasSSE3() && !Subtarget->hasAVX())
    Features[Feature_UseSSE3Bit] = 1;
  if (Subtarget->hasSSSE3())
    Features[Feature_HasSSSE3Bit] = 1;
  if (Subtarget->hasSSSE3() && !Subtarget->hasAVX())
    Features[Feature_UseSSSE3Bit] = 1;
  if (Subtarget->hasSSE41() && !Subtarget->hasAVX())
    Features[Feature_UseSSE41Bit] = 1;
  if (Subtarget->hasSSE42())
    Features[Feature_HasSSE42Bit] = 1;
  if (Subtarget->hasSSE42() && !Subtarget->hasAVX())
    Features[Feature_UseSSE42Bit] = 1;
  if (Subtarget->hasSSE4A())
    Features[Feature_HasSSE4ABit] = 1;
  if (!Subtarget->hasAVX())
    Features[Feature_NoAVXBit] = 1;
  if (Subtarget->hasAVX())
    Features[Feature_HasAVXBit] = 1;
  if (Subtarget->hasAVX2())
    Features[Feature_HasAVX2Bit] = 1;
  if (Subtarget->hasAVX() && !Subtarget->hasAVX2())
    Features[Feature_HasAVX1OnlyBit] = 1;
  if (Subtarget->hasAVX512())
    Features[Feature_HasAVX512Bit] = 1;
  if (Subtarget->hasAVX() && !Subtarget->hasAVX512())
    Features[Feature_UseAVXBit] = 1;
  if (Subtarget->hasAVX2() && !Subtarget->hasAVX512())
    Features[Feature_UseAVX2Bit] = 1;
  if (!Subtarget->hasAVX512())
    Features[Feature_NoAVX512Bit] = 1;
  if (Subtarget->hasCDI())
    Features[Feature_HasCDIBit] = 1;
  if (Subtarget->hasVPOPCNTDQ())
    Features[Feature_HasVPOPCNTDQBit] = 1;
  if (Subtarget->hasERI())
    Features[Feature_HasERIBit] = 1;
  if (Subtarget->hasDQI())
    Features[Feature_HasDQIBit] = 1;
  if (!Subtarget->hasDQI())
    Features[Feature_NoDQIBit] = 1;
  if (Subtarget->hasBWI())
    Features[Feature_HasBWIBit] = 1;
  if (!Subtarget->hasBWI())
    Features[Feature_NoBWIBit] = 1;
  if (Subtarget->hasVLX())
    Features[Feature_HasVLXBit] = 1;
  if (!Subtarget->hasVLX())
    Features[Feature_NoVLXBit] = 1;
  if (!Subtarget->hasVLX() || !Subtarget->hasBWI())
    Features[Feature_NoVLX_Or_NoBWIBit] = 1;
  if (!Subtarget->hasVLX() || !Subtarget->hasDQI())
    Features[Feature_NoVLX_Or_NoDQIBit] = 1;
  if (Subtarget->hasVNNI())
    Features[Feature_HasVNNIBit] = 1;
  if (Subtarget->hasBITALG())
    Features[Feature_HasBITALGBit] = 1;
  if (Subtarget->hasPOPCNT())
    Features[Feature_HasPOPCNTBit] = 1;
  if (Subtarget->hasAES())
    Features[Feature_HasAESBit] = 1;
  if (Subtarget->hasVAES())
    Features[Feature_HasVAESBit] = 1;
  if (!Subtarget->hasVLX() || !Subtarget->hasVAES())
    Features[Feature_NoVLX_Or_NoVAESBit] = 1;
  if (Subtarget->hasFXSR())
    Features[Feature_HasFXSRBit] = 1;
  if (Subtarget->hasXSAVE())
    Features[Feature_HasXSAVEBit] = 1;
  if (Subtarget->hasXSAVEOPT())
    Features[Feature_HasXSAVEOPTBit] = 1;
  if (Subtarget->hasXSAVEC())
    Features[Feature_HasXSAVECBit] = 1;
  if (Subtarget->hasXSAVES())
    Features[Feature_HasXSAVESBit] = 1;
  if (Subtarget->hasPCLMUL())
    Features[Feature_HasPCLMULBit] = 1;
  if (!Subtarget->hasVLX() || !Subtarget->hasVPCLMULQDQ())
    Features[Feature_NoVLX_Or_NoVPCLMULQDQBit] = 1;
  if (Subtarget->hasVPCLMULQDQ())
    Features[Feature_HasVPCLMULQDQBit] = 1;
  if (Subtarget->hasGFNI())
    Features[Feature_HasGFNIBit] = 1;
  if (Subtarget->hasFMA())
    Features[Feature_HasFMABit] = 1;
  if (Subtarget->hasFMA4())
    Features[Feature_HasFMA4Bit] = 1;
  if (!Subtarget->hasFMA4())
    Features[Feature_NoFMA4Bit] = 1;
  if (Subtarget->hasXOP())
    Features[Feature_HasXOPBit] = 1;
  if (Subtarget->hasTBM())
    Features[Feature_HasTBMBit] = 1;
  if (!Subtarget->hasTBM())
    Features[Feature_NoTBMBit] = 1;
  if (Subtarget->hasLWP())
    Features[Feature_HasLWPBit] = 1;
  if (Subtarget->hasMOVBE())
    Features[Feature_HasMOVBEBit] = 1;
  if (Subtarget->hasRDRAND())
    Features[Feature_HasRDRANDBit] = 1;
  if (Subtarget->hasF16C())
    Features[Feature_HasF16CBit] = 1;
  if (Subtarget->hasFSGSBase())
    Features[Feature_HasFSGSBaseBit] = 1;
  if (Subtarget->hasLZCNT())
    Features[Feature_HasLZCNTBit] = 1;
  if (Subtarget->hasBMI())
    Features[Feature_HasBMIBit] = 1;
  if (Subtarget->hasBMI2())
    Features[Feature_HasBMI2Bit] = 1;
  if (!Subtarget->hasBMI2())
    Features[Feature_NoBMI2Bit] = 1;
  if (Subtarget->hasVBMI())
    Features[Feature_HasVBMIBit] = 1;
  if (Subtarget->hasVBMI2())
    Features[Feature_HasVBMI2Bit] = 1;
  if (Subtarget->hasIFMA())
    Features[Feature_HasIFMABit] = 1;
  if (Subtarget->hasRTM())
    Features[Feature_HasRTMBit] = 1;
  if (Subtarget->hasSHA())
    Features[Feature_HasSHABit] = 1;
  if (Subtarget->hasRDSEED())
    Features[Feature_HasRDSEEDBit] = 1;
  if (Subtarget->hasSSEPrefetch())
    Features[Feature_HasSSEPrefetchBit] = 1;
  if (!Subtarget->hasSSEPrefetch())
    Features[Feature_NoSSEPrefetchBit] = 1;
  if (Subtarget->hasPRFCHW())
    Features[Feature_HasPrefetchWBit] = 1;
  if (Subtarget->hasPREFETCHWT1())
    Features[Feature_HasPREFETCHWT1Bit] = 1;
  if (Subtarget->hasLAHFSAHF())
    Features[Feature_HasLAHFSAHFBit] = 1;
  if (Subtarget->hasMWAITX())
    Features[Feature_HasMWAITXBit] = 1;
  if (Subtarget->hasCLZERO())
    Features[Feature_HasCLZEROBit] = 1;
  if (Subtarget->hasCLDEMOTE())
    Features[Feature_HasCLDEMOTEBit] = 1;
  if (Subtarget->hasMOVDIRI())
    Features[Feature_HasMOVDIRIBit] = 1;
  if (Subtarget->hasMOVDIR64B())
    Features[Feature_HasMOVDIR64BBit] = 1;
  if (Subtarget->hasPTWRITE())
    Features[Feature_HasPTWRITEBit] = 1;
  if (!Subtarget->hasSSE1())
    Features[Feature_FPStackf32Bit] = 1;
  if (!Subtarget->hasSSE2())
    Features[Feature_FPStackf64Bit] = 1;
  if (Subtarget->hasCLFLUSHOPT())
    Features[Feature_HasCLFLUSHOPTBit] = 1;
  if (Subtarget->hasCLWB())
    Features[Feature_HasCLWBBit] = 1;
  if (Subtarget->hasWBNOINVD())
    Features[Feature_HasWBNOINVDBit] = 1;
  if (Subtarget->hasRDPID())
    Features[Feature_HasRDPIDBit] = 1;
  if (Subtarget->hasWAITPKG())
    Features[Feature_HasWAITPKGBit] = 1;
  if (Subtarget->hasINVPCID())
    Features[Feature_HasINVPCIDBit] = 1;
  if (Subtarget->hasCmpxchg16b())
    Features[Feature_HasCmpxchg16bBit] = 1;
  if (!Subtarget->is64Bit())
    Features[Feature_Not64BitModeBit] = 1;
  if (Subtarget->is64Bit())
    Features[Feature_In64BitModeBit] = 1;
  if (Subtarget->isTarget64BitLP64())
    Features[Feature_IsLP64Bit] = 1;
  if (!Subtarget->isTarget64BitLP64())
    Features[Feature_NotLP64Bit] = 1;
  if (Subtarget->isTargetPS4())
    Features[Feature_IsPS4Bit] = 1;
  if (!Subtarget->isTargetPS4())
    Features[Feature_NotPS4Bit] = 1;
  if (TM.getCodeModel() == CodeModel::Kernel)
    Features[Feature_KernelCodeBit] = 1;
  if (TM.getCodeModel() == CodeModel::Small ||TM.getCodeModel() == CodeModel::Kernel)
    Features[Feature_NearDataBit] = 1;
  if (!TM.isPositionIndependent())
    Features[Feature_IsNotPICBit] = 1;
  if (Subtarget->isLegalToCallImmediateAddr())
    Features[Feature_CallImmAddrBit] = 1;
  if (!Subtarget->slowTwoMemOps())
    Features[Feature_FavorMemIndirectCallBit] = 1;
  if (Subtarget->hasFastSHLDRotate())
    Features[Feature_HasFastSHLDRotateBit] = 1;
  if (Subtarget->hasMFence())
    Features[Feature_HasMFenceBit] = 1;
  if (Subtarget->useRetpolineIndirectCalls())
    Features[Feature_UseRetpolineIndirectCallsBit] = 1;
  if (!Subtarget->useRetpolineIndirectCalls())
    Features[Feature_NotUseRetpolineIndirectCallsBit] = 1;
  return Features;
}

PredicateBitset X86InstructionSelector::
computeAvailableFunctionFeatures(const X86Subtarget *Subtarget, const MachineFunction *MF) const {
  PredicateBitset Features;
  if (!Subtarget->isTargetWin64() ||Subtarget->getFrameLowering()->hasFP(*MF))
    Features[Feature_NotWin64WithoutFPBit] = 1;
  if (MF->getFunction().optForSize())
    Features[Feature_OptForSizeBit] = 1;
  if (MF->getFunction().optForMinSize())
    Features[Feature_OptForMinSizeBit] = 1;
  if (!MF->getFunction().optForSize())
    Features[Feature_OptForSpeedBit] = 1;
  if (!Subtarget->slowIncDec() || MF->getFunction().optForSize())
    Features[Feature_UseIncDecBit] = 1;
  if (MF->getFunction().optForSize() || !Subtarget->hasSSE41())
    Features[Feature_NoSSE41_Or_OptForSizeBit] = 1;
  return Features;
}

// LLT Objects.
enum {
  GILLT_s1,
  GILLT_s8,
  GILLT_s16,
  GILLT_s32,
  GILLT_s64,
  GILLT_s80,
  GILLT_s128,
  GILLT_v2s1,
  GILLT_v2s64,
  GILLT_v4s1,
  GILLT_v4s32,
  GILLT_v4s64,
  GILLT_v8s1,
  GILLT_v8s16,
  GILLT_v8s32,
  GILLT_v8s64,
  GILLT_v16s1,
  GILLT_v16s8,
  GILLT_v16s16,
  GILLT_v16s32,
  GILLT_v32s1,
  GILLT_v32s8,
  GILLT_v32s16,
  GILLT_v64s1,
  GILLT_v64s8,
};
const static size_t NumTypeObjects = 25;
const static LLT TypeObjects[] = {
  LLT::scalar(1),
  LLT::scalar(8),
  LLT::scalar(16),
  LLT::scalar(32),
  LLT::scalar(64),
  LLT::scalar(80),
  LLT::scalar(128),
  LLT::vector(2, 1),
  LLT::vector(2, 64),
  LLT::vector(4, 1),
  LLT::vector(4, 32),
  LLT::vector(4, 64),
  LLT::vector(8, 1),
  LLT::vector(8, 16),
  LLT::vector(8, 32),
  LLT::vector(8, 64),
  LLT::vector(16, 1),
  LLT::vector(16, 8),
  LLT::vector(16, 16),
  LLT::vector(16, 32),
  LLT::vector(32, 1),
  LLT::vector(32, 8),
  LLT::vector(32, 16),
  LLT::vector(64, 1),
  LLT::vector(64, 8),
};

// Feature bitsets.
enum {
  GIFBS_Invalid,
  GIFBS_FPStackf32,
  GIFBS_FPStackf64,
  GIFBS_Has3DNow,
  GIFBS_HasAVX,
  GIFBS_HasAVX1Only,
  GIFBS_HasAVX2,
  GIFBS_HasAVX512,
  GIFBS_HasBITALG,
  GIFBS_HasBMI,
  GIFBS_HasBMI2,
  GIFBS_HasBWI,
  GIFBS_HasCDI,
  GIFBS_HasDQI,
  GIFBS_HasLWP,
  GIFBS_HasMFence,
  GIFBS_HasMMX,
  GIFBS_HasMOVBE,
  GIFBS_HasPTWRITE,
  GIFBS_HasRTM,
  GIFBS_HasSHA,
  GIFBS_HasSSE1,
  GIFBS_HasSSE2,
  GIFBS_HasSSE42,
  GIFBS_HasSSE4A,
  GIFBS_HasTBM,
  GIFBS_HasVLX,
  GIFBS_HasVPOPCNTDQ,
  GIFBS_HasWAITPKG,
  GIFBS_HasWBNOINVD,
  GIFBS_HasXOP,
  GIFBS_In64BitMode,
  GIFBS_NoDQI,
  GIFBS_Not64BitMode,
  GIFBS_UseAVX,
  GIFBS_UseIncDec,
  GIFBS_UseSSE1,
  GIFBS_UseSSE2,
  GIFBS_UseSSE41,
  GIFBS_UseSSSE3,
  GIFBS_HasAES_HasAVX,
  GIFBS_HasAES_NoAVX,
  GIFBS_HasAVX_NoVLX,
  GIFBS_HasAVX_NoVLX_Or_NoBWI,
  GIFBS_HasAVX2_NoVLX,
  GIFBS_HasAVX2_NoVLX_Or_NoBWI,
  GIFBS_HasAVX512_HasVAES,
  GIFBS_HasAVX512_HasVLX,
  GIFBS_HasAVX512_HasVPCLMULQDQ,
  GIFBS_HasBITALG_HasVLX,
  GIFBS_HasBWI_HasVLX,
  GIFBS_HasCDI_HasVLX,
  GIFBS_HasDQI_HasVLX,
  GIFBS_HasDQI_NoBWI,
  GIFBS_HasFSGSBase_In64BitMode,
  GIFBS_HasPCLMUL_NoAVX,
  GIFBS_HasPTWRITE_In64BitMode,
  GIFBS_HasRDPID_Not64BitMode,
  GIFBS_HasSSE2_NoAVX512,
  GIFBS_HasVAES_HasVLX,
  GIFBS_HasVAES_NoVLX,
  GIFBS_HasVLX_HasVPCLMULQDQ,
  GIFBS_HasVLX_HasVPOPCNTDQ,
  GIFBS_HasVPCLMULQDQ_NoVLX,
  GIFBS_HasWAITPKG_In64BitMode,
  GIFBS_HasWAITPKG_Not64BitMode,
  GIFBS_Not64BitMode_OptForSize,
  GIFBS_NotWin64WithoutFP_OptForMinSize,
  GIFBS_HasAES_HasAVX_NoVLX_Or_NoVAES,
  GIFBS_HasAVX_HasPCLMUL_NoVLX_Or_NoVPCLMULQDQ,
  GIFBS_HasDQI_HasVLX_NoBWI,
};
const static PredicateBitset FeatureBitsets[] {
  {}, // GIFBS_Invalid
  {Feature_FPStackf32Bit, },
  {Feature_FPStackf64Bit, },
  {Feature_Has3DNowBit, },
  {Feature_HasAVXBit, },
  {Feature_HasAVX1OnlyBit, },
  {Feature_HasAVX2Bit, },
  {Feature_HasAVX512Bit, },
  {Feature_HasBITALGBit, },
  {Feature_HasBMIBit, },
  {Feature_HasBMI2Bit, },
  {Feature_HasBWIBit, },
  {Feature_HasCDIBit, },
  {Feature_HasDQIBit, },
  {Feature_HasLWPBit, },
  {Feature_HasMFenceBit, },
  {Feature_HasMMXBit, },
  {Feature_HasMOVBEBit, },
  {Feature_HasPTWRITEBit, },
  {Feature_HasRTMBit, },
  {Feature_HasSHABit, },
  {Feature_HasSSE1Bit, },
  {Feature_HasSSE2Bit, },
  {Feature_HasSSE42Bit, },
  {Feature_HasSSE4ABit, },
  {Feature_HasTBMBit, },
  {Feature_HasVLXBit, },
  {Feature_HasVPOPCNTDQBit, },
  {Feature_HasWAITPKGBit, },
  {Feature_HasWBNOINVDBit, },
  {Feature_HasXOPBit, },
  {Feature_In64BitModeBit, },
  {Feature_NoDQIBit, },
  {Feature_Not64BitModeBit, },
  {Feature_UseAVXBit, },
  {Feature_UseIncDecBit, },
  {Feature_UseSSE1Bit, },
  {Feature_UseSSE2Bit, },
  {Feature_UseSSE41Bit, },
  {Feature_UseSSSE3Bit, },
  {Feature_HasAESBit, Feature_HasAVXBit, },
  {Feature_HasAESBit, Feature_NoAVXBit, },
  {Feature_HasAVXBit, Feature_NoVLXBit, },
  {Feature_HasAVXBit, Feature_NoVLX_Or_NoBWIBit, },
  {Feature_HasAVX2Bit, Feature_NoVLXBit, },
  {Feature_HasAVX2Bit, Feature_NoVLX_Or_NoBWIBit, },
  {Feature_HasAVX512Bit, Feature_HasVAESBit, },
  {Feature_HasAVX512Bit, Feature_HasVLXBit, },
  {Feature_HasAVX512Bit, Feature_HasVPCLMULQDQBit, },
  {Feature_HasBITALGBit, Feature_HasVLXBit, },
  {Feature_HasBWIBit, Feature_HasVLXBit, },
  {Feature_HasCDIBit, Feature_HasVLXBit, },
  {Feature_HasDQIBit, Feature_HasVLXBit, },
  {Feature_HasDQIBit, Feature_NoBWIBit, },
  {Feature_HasFSGSBaseBit, Feature_In64BitModeBit, },
  {Feature_HasPCLMULBit, Feature_NoAVXBit, },
  {Feature_HasPTWRITEBit, Feature_In64BitModeBit, },
  {Feature_HasRDPIDBit, Feature_Not64BitModeBit, },
  {Feature_HasSSE2Bit, Feature_NoAVX512Bit, },
  {Feature_HasVAESBit, Feature_HasVLXBit, },
  {Feature_HasVAESBit, Feature_NoVLXBit, },
  {Feature_HasVLXBit, Feature_HasVPCLMULQDQBit, },
  {Feature_HasVLXBit, Feature_HasVPOPCNTDQBit, },
  {Feature_HasVPCLMULQDQBit, Feature_NoVLXBit, },
  {Feature_HasWAITPKGBit, Feature_In64BitModeBit, },
  {Feature_HasWAITPKGBit, Feature_Not64BitModeBit, },
  {Feature_Not64BitModeBit, Feature_OptForSizeBit, },
  {Feature_NotWin64WithoutFPBit, Feature_OptForMinSizeBit, },
  {Feature_HasAESBit, Feature_HasAVXBit, Feature_NoVLX_Or_NoVAESBit, },
  {Feature_HasAVXBit, Feature_HasPCLMULBit, Feature_NoVLX_Or_NoVPCLMULQDQBit, },
  {Feature_HasDQIBit, Feature_HasVLXBit, Feature_NoBWIBit, },
};

// ComplexPattern predicates.
enum {
  GICP_Invalid,
};
// See constructor for table contents

// PatFrag predicates.
enum {
  GIPFP_I64_Predicate_AndMask64 = GIPFP_I64_Invalid + 1,
  GIPFP_I64_Predicate_BTCBTSMask64,
  GIPFP_I64_Predicate_BTRMask64,
  GIPFP_I64_Predicate_PrefetchWT1Level,
  GIPFP_I64_Predicate_i16immSExt8,
  GIPFP_I64_Predicate_i32immSExt8,
  GIPFP_I64_Predicate_i64immSExt32,
  GIPFP_I64_Predicate_i64immSExt8,
  GIPFP_I64_Predicate_i64immZExt32,
  GIPFP_I64_Predicate_i64immZExt32SExt8,
  GIPFP_I64_Predicate_immShift16,
  GIPFP_I64_Predicate_immShift32,
  GIPFP_I64_Predicate_immShift64,
  GIPFP_I64_Predicate_immShift8,
  GIPFP_I64_Predicate_immff00_ffff,
};
bool X86InstructionSelector::testImmPredicate_I64(unsigned PredicateID, int64_t Imm) const {
  switch (PredicateID) {
  case GIPFP_I64_Predicate_AndMask64: {
    
  return isMask_64(Imm) && !isUInt<32>(Imm);

    llvm_unreachable("ImmediateCode should have returned");
    return false;
  }
  case GIPFP_I64_Predicate_BTCBTSMask64: {
    
  return !isInt<32>(Imm) && isPowerOf2_64(Imm);

    llvm_unreachable("ImmediateCode should have returned");
    return false;
  }
  case GIPFP_I64_Predicate_BTRMask64: {
    
  return !isUInt<32>(Imm) && !isInt<32>(Imm) && isPowerOf2_64(~Imm);

    llvm_unreachable("ImmediateCode should have returned");
    return false;
  }
  case GIPFP_I64_Predicate_PrefetchWT1Level: {
    
  return Imm < 3;

    llvm_unreachable("ImmediateCode should have returned");
    return false;
  }
  case GIPFP_I64_Predicate_i16immSExt8: {
     return isInt<8>(Imm); 
    llvm_unreachable("ImmediateCode should have returned");
    return false;
  }
  case GIPFP_I64_Predicate_i32immSExt8: {
     return isInt<8>(Imm); 
    llvm_unreachable("ImmediateCode should have returned");
    return false;
  }
  case GIPFP_I64_Predicate_i64immSExt32: {
     return isInt<32>(Imm); 
    llvm_unreachable("ImmediateCode should have returned");
    return false;
  }
  case GIPFP_I64_Predicate_i64immSExt8: {
     return isInt<8>(Imm); 
    llvm_unreachable("ImmediateCode should have returned");
    return false;
  }
  case GIPFP_I64_Predicate_i64immZExt32: {
     return isUInt<32>(Imm); 
    llvm_unreachable("ImmediateCode should have returned");
    return false;
  }
  case GIPFP_I64_Predicate_i64immZExt32SExt8: {
    
  return isUInt<32>(Imm) && isInt<8>(static_cast<int32_t>(Imm));

    llvm_unreachable("ImmediateCode should have returned");
    return false;
  }
  case GIPFP_I64_Predicate_immShift16: {
    
  return countTrailingOnes<uint64_t>(Imm) >= 4;

    llvm_unreachable("ImmediateCode should have returned");
    return false;
  }
  case GIPFP_I64_Predicate_immShift32: {
    
  return countTrailingOnes<uint64_t>(Imm) >= 5;

    llvm_unreachable("ImmediateCode should have returned");
    return false;
  }
  case GIPFP_I64_Predicate_immShift64: {
    
  return countTrailingOnes<uint64_t>(Imm) >= 6;

    llvm_unreachable("ImmediateCode should have returned");
    return false;
  }
  case GIPFP_I64_Predicate_immShift8: {
    
  return countTrailingOnes<uint64_t>(Imm) >= 3;

    llvm_unreachable("ImmediateCode should have returned");
    return false;
  }
  case GIPFP_I64_Predicate_immff00_ffff: {
    
  return Imm >= 0xff00 && Imm <= 0xffff;

    llvm_unreachable("ImmediateCode should have returned");
    return false;
  }
  }
  llvm_unreachable("Unknown predicate");
  return false;
}
// PatFrag predicates.
enum {
  GIPFP_APFloat_Predicate_fpimm0 = GIPFP_APFloat_Invalid + 1,
  GIPFP_APFloat_Predicate_fpimm1,
  GIPFP_APFloat_Predicate_fpimmneg0,
  GIPFP_APFloat_Predicate_fpimmneg1,
};
bool X86InstructionSelector::testImmPredicate_APFloat(unsigned PredicateID, const APFloat & Imm) const {
  switch (PredicateID) {
  case GIPFP_APFloat_Predicate_fpimm0: {
    
  return Imm.isExactlyValue(+0.0);

    llvm_unreachable("ImmediateCode should have returned");
    return false;
  }
  case GIPFP_APFloat_Predicate_fpimm1: {
    
  return Imm.isExactlyValue(+1.0);

    llvm_unreachable("ImmediateCode should have returned");
    return false;
  }
  case GIPFP_APFloat_Predicate_fpimmneg0: {
    
  return Imm.isExactlyValue(-0.0);

    llvm_unreachable("ImmediateCode should have returned");
    return false;
  }
  case GIPFP_APFloat_Predicate_fpimmneg1: {
    
  return Imm.isExactlyValue(-1.0);

    llvm_unreachable("ImmediateCode should have returned");
    return false;
  }
  }
  llvm_unreachable("Unknown predicate");
  return false;
}
bool X86InstructionSelector::testImmPredicate_APInt(unsigned PredicateID, const APInt & Imm) const {
  llvm_unreachable("Unknown predicate");
  return false;
}
bool X86InstructionSelector::testMIPredicate_MI(unsigned PredicateID, const MachineInstr & MI) const {
  const MachineFunction &MF = *MI.getParent()->getParent();
  const MachineRegisterInfo &MRI = MF.getRegInfo();
  (void)MRI;
  llvm_unreachable("Unknown predicate");
  return false;
}

X86InstructionSelector::ComplexMatcherMemFn
X86InstructionSelector::ComplexPredicateFns[] = {
  nullptr, // GICP_Invalid
};

// Custom renderers.
enum {
  GICR_Invalid,
};
X86InstructionSelector::CustomRendererFn
X86InstructionSelector::CustomRenderers[] = {
  nullptr, // GICP_Invalid
};

bool X86InstructionSelector::selectImpl(MachineInstr &I, CodeGenCoverage &CoverageInfo) const {
  MachineFunction &MF = *I.getParent()->getParent();
  MachineRegisterInfo &MRI = MF.getRegInfo();
  // FIXME: This should be computed on a per-function basis rather than per-insn.
  AvailableFunctionFeatures = computeAvailableFunctionFeatures(&STI, &MF);
  const PredicateBitset AvailableFeatures = getAvailableFeatures();
  NewMIVector OutMIs;
  State.MIs.clear();
  State.MIs.push_back(&I);

  if (executeMatchTable(*this, OutMIs, State, ISelInfo, getMatchTable(), TII, MRI, TRI, RBI, AvailableFeatures, CoverageInfo)) {
    return true;
  }

  return false;
}

const int64_t *X86InstructionSelector::getMatchTable() const {
  constexpr static int64_t MatchTable0[] = {
    GIM_SwitchOpcode, /*MI*/0, /*[*/34, 135, /*)*//*default:*//*Label 34*/ 29511,
    /*TargetOpcode::G_ADD*//*Label 0*/ 106,
    /*TargetOpcode::G_SUB*//*Label 1*/ 1723,
    /*TargetOpcode::G_MUL*//*Label 2*/ 2804, 0, 0, 0, 0,
    /*TargetOpcode::G_AND*//*Label 3*/ 3570,
    /*TargetOpcode::G_OR*//*Label 4*/ 6682,
    /*TargetOpcode::G_XOR*//*Label 5*/ 9353, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    /*TargetOpcode::G_BITCAST*//*Label 6*/ 11326, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    /*TargetOpcode::G_INTRINSIC*//*Label 7*/ 11726,
    /*TargetOpcode::G_INTRINSIC_W_SIDE_EFFECTS*//*Label 8*/ 18309,
    /*TargetOpcode::G_ANYEXT*//*Label 9*/ 19740,
    /*TargetOpcode::G_TRUNC*//*Label 10*/ 19856,
    /*TargetOpcode::G_CONSTANT*//*Label 11*/ 20191,
    /*TargetOpcode::G_FCONSTANT*//*Label 12*/ 20394, 0, 0,
    /*TargetOpcode::G_SEXT*//*Label 13*/ 20587,
    /*TargetOpcode::G_ZEXT*//*Label 14*/ 21130,
    /*TargetOpcode::G_SHL*//*Label 15*/ 21265,
    /*TargetOpcode::G_LSHR*//*Label 16*/ 21990,
    /*TargetOpcode::G_ASHR*//*Label 17*/ 22821, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    /*TargetOpcode::G_FADD*//*Label 18*/ 23478,
    /*TargetOpcode::G_FSUB*//*Label 19*/ 24091,
    /*TargetOpcode::G_FMUL*//*Label 20*/ 24704, 0,
    /*TargetOpcode::G_FDIV*//*Label 21*/ 25317, 0, 0, 0, 0, 0, 0,
    /*TargetOpcode::G_FNEG*//*Label 22*/ 25930,
    /*TargetOpcode::G_FPEXT*//*Label 23*/ 26019,
    /*TargetOpcode::G_FPTRUNC*//*Label 24*/ 26295,
    /*TargetOpcode::G_FPTOSI*//*Label 25*/ 26580,
    /*TargetOpcode::G_FPTOUI*//*Label 26*/ 27210,
    /*TargetOpcode::G_SITOFP*//*Label 27*/ 27564,
    /*TargetOpcode::G_UITOFP*//*Label 28*/ 28394, 0, 0, 0,
    /*TargetOpcode::G_BR*//*Label 29*/ 28848, 0, 0, 0, 0,
    /*TargetOpcode::G_CTTZ_ZERO_UNDEF*//*Label 30*/ 28861,
    /*TargetOpcode::G_CTLZ*//*Label 31*/ 28946, 0,
    /*TargetOpcode::G_CTPOP*//*Label 32*/ 29109,
    /*TargetOpcode::G_BSWAP*//*Label 33*/ 29421,
    // Label 0: @106
    GIM_SwitchType, /*MI*/0, /*Op*/0, /*[*/1, 25, /*)*//*default:*//*Label 51*/ 1722,
    /*GILLT_s8*//*Label 35*/ 136,
    /*GILLT_s16*//*Label 36*/ 250,
    /*GILLT_s32*//*Label 37*/ 422,
    /*GILLT_s64*//*Label 38*/ 594, 0, 0, 0,
    /*GILLT_v2s64*//*Label 39*/ 794, 0,
    /*GILLT_v4s32*//*Label 40*/ 875,
    /*GILLT_v4s64*//*Label 41*/ 1078, 0,
    /*GILLT_v8s16*//*Label 42*/ 1136,
    /*GILLT_v8s32*//*Label 43*/ 1339,
    /*GILLT_v8s64*//*Label 44*/ 1397, 0,
    /*GILLT_v16s8*//*Label 45*/ 1429,
    /*GILLT_v16s16*//*Label 46*/ 1510,
    /*GILLT_v16s32*//*Label 47*/ 1568, 0,
    /*GILLT_v32s8*//*Label 48*/ 1600,
    /*GILLT_v32s16*//*Label 49*/ 1658, 0,
    /*GILLT_v64s8*//*Label 50*/ 1690,
    // Label 35: @136
    GIM_Try, /*On fail goto*//*Label 52*/ 249,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_s8,
      GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_s8,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::GR8RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::GR8RegClassID,
      GIM_Try, /*On fail goto*//*Label 53*/ 178, // Rule ID 16148 //
        GIM_CheckFeatures, GIFBS_UseIncDec,
        GIM_CheckConstantInt, /*MI*/0, /*Op*/2, 1,
        // (add:{ *:[i8] } GR8:{ *:[i8] }:$src, 1:{ *:[i8] })  =>  (INC8r:{ *:[i8] }:{ *:[i32] } GR8:{ *:[i8] }:$src)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::INC8r,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // src
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 16148,
        GIR_Done,
      // Label 53: @178
      GIM_Try, /*On fail goto*//*Label 54*/ 202, // Rule ID 16152 //
        GIM_CheckFeatures, GIFBS_UseIncDec,
        GIM_CheckConstantInt, /*MI*/0, /*Op*/2, -1,
        // (add:{ *:[i8] } GR8:{ *:[i8] }:$src, -1:{ *:[i8] })  =>  (DEC8r:{ *:[i8] }:{ *:[i32] } GR8:{ *:[i8] }:$src)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::DEC8r,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // src
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 16152,
        GIR_Done,
      // Label 54: @202
      GIM_Try, /*On fail goto*//*Label 55*/ 232, // Rule ID 16102 //
        GIM_RecordInsn, /*DefineMI*/1, /*MI*/0, /*OpIdx*/2, // MIs[1]
        GIM_CheckOpcode, /*MI*/1, TargetOpcode::G_CONSTANT,
        // MIs[1] Operand 1
        // No operand predicates
        GIM_CheckIsSafeToFold, /*InsnID*/1,
        // (add:{ *:[i8] } GR8:{ *:[i8] }:$src1, (imm:{ *:[i8] }):$src2)  =>  (ADD8ri:{ *:[i8] }:{ *:[i32] } GR8:{ *:[i8] }:$src1, (imm:{ *:[i8] }):$src2)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::ADD8ri,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // src1
        GIR_CopyConstantAsSImm, /*NewInsnID*/0, /*OldInsnID*/1, // src2
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 16102,
        GIR_Done,
      // Label 55: @232
      GIM_Try, /*On fail goto*//*Label 56*/ 248, // Rule ID 16094 //
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::GR8RegClassID,
        // (add:{ *:[i8] } GR8:{ *:[i8] }:$src1, GR8:{ *:[i8] }:$src2)  =>  (ADD8rr:{ *:[i8] }:{ *:[i32] } GR8:{ *:[i8] }:$src1, GR8:{ *:[i8] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::ADD8rr,
        GIR_AddImplicitDef, /*InsnID*/0, X86::EFLAGS,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 16094,
        GIR_Done,
      // Label 56: @248
      GIM_Reject,
    // Label 52: @249
    GIM_Reject,
    // Label 36: @250
    GIM_Try, /*On fail goto*//*Label 57*/ 421,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_s16,
      GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_s16,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::GR16RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::GR16RegClassID,
      GIM_Try, /*On fail goto*//*Label 58*/ 293, // Rule ID 15960 //
        GIM_CheckConstantInt, /*MI*/0, /*Op*/2, 128,
        // (add:{ *:[i16] } GR16:{ *:[i16] }:$src1, 128:{ *:[i16] })  =>  (SUB16ri8:{ *:[i16] }:{ *:[i32] } GR16:{ *:[i16] }:$src1, -128:{ *:[i16] })
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::SUB16ri8,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // src1
        GIR_AddImm, /*InsnID*/0, /*Imm*/-128,
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 15960,
        GIR_Done,
      // Label 58: @293
      GIM_Try, /*On fail goto*//*Label 59*/ 317, // Rule ID 16149 //
        GIM_CheckFeatures, GIFBS_UseIncDec,
        GIM_CheckConstantInt, /*MI*/0, /*Op*/2, 1,
        // (add:{ *:[i16] } GR16:{ *:[i16] }:$src, 1:{ *:[i16] })  =>  (INC16r:{ *:[i16] }:{ *:[i32] } GR16:{ *:[i16] }:$src)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::INC16r,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // src
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 16149,
        GIR_Done,
      // Label 59: @317
      GIM_Try, /*On fail goto*//*Label 60*/ 341, // Rule ID 16153 //
        GIM_CheckFeatures, GIFBS_UseIncDec,
        GIM_CheckConstantInt, /*MI*/0, /*Op*/2, -1,
        // (add:{ *:[i16] } GR16:{ *:[i16] }:$src, -1:{ *:[i16] })  =>  (DEC16r:{ *:[i16] }:{ *:[i32] } GR16:{ *:[i16] }:$src)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::DEC16r,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // src
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 16153,
        GIR_Done,
      // Label 60: @341
      GIM_Try, /*On fail goto*//*Label 61*/ 374, // Rule ID 16105 //
        GIM_RecordInsn, /*DefineMI*/1, /*MI*/0, /*OpIdx*/2, // MIs[1]
        GIM_CheckOpcode, /*MI*/1, TargetOpcode::G_CONSTANT,
        GIM_CheckI64ImmPredicate, /*MI*/1, /*Predicate*/GIPFP_I64_Predicate_i16immSExt8,
        // MIs[1] Operand 1
        // No operand predicates
        GIM_CheckIsSafeToFold, /*InsnID*/1,
        // (add:{ *:[i16] } GR16:{ *:[i16] }:$src1, (imm:{ *:[i16] })<<P:Predicate_i16immSExt8>>:$src2)  =>  (ADD16ri8:{ *:[i16] }:{ *:[i32] } GR16:{ *:[i16] }:$src1, (imm:{ *:[i16] })<<P:Predicate_i16immSExt8>>:$src2)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::ADD16ri8,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // src1
        GIR_CopyConstantAsSImm, /*NewInsnID*/0, /*OldInsnID*/1, // src2
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 16105,
        GIR_Done,
      // Label 61: @374
      GIM_Try, /*On fail goto*//*Label 62*/ 404, // Rule ID 16103 //
        GIM_RecordInsn, /*DefineMI*/1, /*MI*/0, /*OpIdx*/2, // MIs[1]
        GIM_CheckOpcode, /*MI*/1, TargetOpcode::G_CONSTANT,
        // MIs[1] Operand 1
        // No operand predicates
        GIM_CheckIsSafeToFold, /*InsnID*/1,
        // (add:{ *:[i16] } GR16:{ *:[i16] }:$src1, (imm:{ *:[i16] }):$src2)  =>  (ADD16ri:{ *:[i16] }:{ *:[i32] } GR16:{ *:[i16] }:$src1, (imm:{ *:[i16] }):$src2)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::ADD16ri,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // src1
        GIR_CopyConstantAsSImm, /*NewInsnID*/0, /*OldInsnID*/1, // src2
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 16103,
        GIR_Done,
      // Label 62: @404
      GIM_Try, /*On fail goto*//*Label 63*/ 420, // Rule ID 16095 //
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::GR16RegClassID,
        // (add:{ *:[i16] } GR16:{ *:[i16] }:$src1, GR16:{ *:[i16] }:$src2)  =>  (ADD16rr:{ *:[i16] }:{ *:[i32] } GR16:{ *:[i16] }:$src1, GR16:{ *:[i16] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::ADD16rr,
        GIR_AddImplicitDef, /*InsnID*/0, X86::EFLAGS,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 16095,
        GIR_Done,
      // Label 63: @420
      GIM_Reject,
    // Label 57: @421
    GIM_Reject,
    // Label 37: @422
    GIM_Try, /*On fail goto*//*Label 64*/ 593,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_s32,
      GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_s32,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::GR32RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::GR32RegClassID,
      GIM_Try, /*On fail goto*//*Label 65*/ 465, // Rule ID 15962 //
        GIM_CheckConstantInt, /*MI*/0, /*Op*/2, 128,
        // (add:{ *:[i32] } GR32:{ *:[i32] }:$src1, 128:{ *:[i32] })  =>  (SUB32ri8:{ *:[i32] }:{ *:[i32] } GR32:{ *:[i32] }:$src1, -128:{ *:[i32] })
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::SUB32ri8,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // src1
        GIR_AddImm, /*InsnID*/0, /*Imm*/-128,
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 15962,
        GIR_Done,
      // Label 65: @465
      GIM_Try, /*On fail goto*//*Label 66*/ 489, // Rule ID 16150 //
        GIM_CheckFeatures, GIFBS_UseIncDec,
        GIM_CheckConstantInt, /*MI*/0, /*Op*/2, 1,
        // (add:{ *:[i32] } GR32:{ *:[i32] }:$src, 1:{ *:[i32] })  =>  (INC32r:{ *:[i32] }:{ *:[i32] } GR32:{ *:[i32] }:$src)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::INC32r,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // src
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 16150,
        GIR_Done,
      // Label 66: @489
      GIM_Try, /*On fail goto*//*Label 67*/ 513, // Rule ID 16154 //
        GIM_CheckFeatures, GIFBS_UseIncDec,
        GIM_CheckConstantInt, /*MI*/0, /*Op*/2, -1,
        // (add:{ *:[i32] } GR32:{ *:[i32] }:$src, -1:{ *:[i32] })  =>  (DEC32r:{ *:[i32] }:{ *:[i32] } GR32:{ *:[i32] }:$src)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::DEC32r,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // src
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 16154,
        GIR_Done,
      // Label 67: @513
      GIM_Try, /*On fail goto*//*Label 68*/ 546, // Rule ID 16106 //
        GIM_RecordInsn, /*DefineMI*/1, /*MI*/0, /*OpIdx*/2, // MIs[1]
        GIM_CheckOpcode, /*MI*/1, TargetOpcode::G_CONSTANT,
        GIM_CheckI64ImmPredicate, /*MI*/1, /*Predicate*/GIPFP_I64_Predicate_i32immSExt8,
        // MIs[1] Operand 1
        // No operand predicates
        GIM_CheckIsSafeToFold, /*InsnID*/1,
        // (add:{ *:[i32] } GR32:{ *:[i32] }:$src1, (imm:{ *:[i32] })<<P:Predicate_i32immSExt8>>:$src2)  =>  (ADD32ri8:{ *:[i32] }:{ *:[i32] } GR32:{ *:[i32] }:$src1, (imm:{ *:[i32] })<<P:Predicate_i32immSExt8>>:$src2)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::ADD32ri8,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // src1
        GIR_CopyConstantAsSImm, /*NewInsnID*/0, /*OldInsnID*/1, // src2
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 16106,
        GIR_Done,
      // Label 68: @546
      GIM_Try, /*On fail goto*//*Label 69*/ 576, // Rule ID 16104 //
        GIM_RecordInsn, /*DefineMI*/1, /*MI*/0, /*OpIdx*/2, // MIs[1]
        GIM_CheckOpcode, /*MI*/1, TargetOpcode::G_CONSTANT,
        // MIs[1] Operand 1
        // No operand predicates
        GIM_CheckIsSafeToFold, /*InsnID*/1,
        // (add:{ *:[i32] } GR32:{ *:[i32] }:$src1, (imm:{ *:[i32] }):$src2)  =>  (ADD32ri:{ *:[i32] }:{ *:[i32] } GR32:{ *:[i32] }:$src1, (imm:{ *:[i32] }):$src2)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::ADD32ri,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // src1
        GIR_CopyConstantAsSImm, /*NewInsnID*/0, /*OldInsnID*/1, // src2
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 16104,
        GIR_Done,
      // Label 69: @576
      GIM_Try, /*On fail goto*//*Label 70*/ 592, // Rule ID 16096 //
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::GR32RegClassID,
        // (add:{ *:[i32] } GR32:{ *:[i32] }:$src1, GR32:{ *:[i32] }:$src2)  =>  (ADD32rr:{ *:[i32] }:{ *:[i32] } GR32:{ *:[i32] }:$src1, GR32:{ *:[i32] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::ADD32rr,
        GIR_AddImplicitDef, /*InsnID*/0, X86::EFLAGS,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 16096,
        GIR_Done,
      // Label 70: @592
      GIM_Reject,
    // Label 64: @593
    GIM_Reject,
    // Label 38: @594
    GIM_Try, /*On fail goto*//*Label 71*/ 793,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_s64,
      GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_s64,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::GR64RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::GR64RegClassID,
      GIM_Try, /*On fail goto*//*Label 72*/ 637, // Rule ID 15964 //
        GIM_CheckConstantInt, /*MI*/0, /*Op*/2, 128,
        // (add:{ *:[i64] } GR64:{ *:[i64] }:$src1, 128:{ *:[i64] })  =>  (SUB64ri8:{ *:[i64] }:{ *:[i32] } GR64:{ *:[i64] }:$src1, -128:{ *:[i64] })
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::SUB64ri8,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // src1
        GIR_AddImm, /*InsnID*/0, /*Imm*/-128,
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 15964,
        GIR_Done,
      // Label 72: @637
      GIM_Try, /*On fail goto*//*Label 73*/ 662, // Rule ID 15966 //
        GIM_CheckConstantInt, /*MI*/0, /*Op*/2, 2147483648,
        // (add:{ *:[i64] } GR64:{ *:[i64] }:$src1, 2147483648:{ *:[i64] })  =>  (SUB64ri32:{ *:[i64] }:{ *:[i32] } GR64:{ *:[i64] }:$src1, -2147483648:{ *:[i64] })
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::SUB64ri32,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // src1
        GIR_AddImm, /*InsnID*/0, /*Imm*/-2147483648,
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 15966,
        GIR_Done,
      // Label 73: @662
      GIM_Try, /*On fail goto*//*Label 74*/ 686, // Rule ID 16151 //
        GIM_CheckFeatures, GIFBS_UseIncDec,
        GIM_CheckConstantInt, /*MI*/0, /*Op*/2, 1,
        // (add:{ *:[i64] } GR64:{ *:[i64] }:$src, 1:{ *:[i64] })  =>  (INC64r:{ *:[i64] }:{ *:[i32] } GR64:{ *:[i64] }:$src)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::INC64r,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // src
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 16151,
        GIR_Done,
      // Label 74: @686
      GIM_Try, /*On fail goto*//*Label 75*/ 710, // Rule ID 16155 //
        GIM_CheckFeatures, GIFBS_UseIncDec,
        GIM_CheckConstantInt, /*MI*/0, /*Op*/2, -1,
        // (add:{ *:[i64] } GR64:{ *:[i64] }:$src, -1:{ *:[i64] })  =>  (DEC64r:{ *:[i64] }:{ *:[i32] } GR64:{ *:[i64] }:$src)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::DEC64r,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // src
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 16155,
        GIR_Done,
      // Label 75: @710
      GIM_Try, /*On fail goto*//*Label 76*/ 743, // Rule ID 16107 //
        GIM_RecordInsn, /*DefineMI*/1, /*MI*/0, /*OpIdx*/2, // MIs[1]
        GIM_CheckOpcode, /*MI*/1, TargetOpcode::G_CONSTANT,
        GIM_CheckI64ImmPredicate, /*MI*/1, /*Predicate*/GIPFP_I64_Predicate_i64immSExt8,
        // MIs[1] Operand 1
        // No operand predicates
        GIM_CheckIsSafeToFold, /*InsnID*/1,
        // (add:{ *:[i64] } GR64:{ *:[i64] }:$src1, (imm:{ *:[i64] })<<P:Predicate_i64immSExt8>>:$src2)  =>  (ADD64ri8:{ *:[i64] }:{ *:[i32] } GR64:{ *:[i64] }:$src1, (imm:{ *:[i64] })<<P:Predicate_i64immSExt8>>:$src2)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::ADD64ri8,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // src1
        GIR_CopyConstantAsSImm, /*NewInsnID*/0, /*OldInsnID*/1, // src2
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 16107,
        GIR_Done,
      // Label 76: @743
      GIM_Try, /*On fail goto*//*Label 77*/ 776, // Rule ID 16108 //
        GIM_RecordInsn, /*DefineMI*/1, /*MI*/0, /*OpIdx*/2, // MIs[1]
        GIM_CheckOpcode, /*MI*/1, TargetOpcode::G_CONSTANT,
        GIM_CheckI64ImmPredicate, /*MI*/1, /*Predicate*/GIPFP_I64_Predicate_i64immSExt32,
        // MIs[1] Operand 1
        // No operand predicates
        GIM_CheckIsSafeToFold, /*InsnID*/1,
        // (add:{ *:[i64] } GR64:{ *:[i64] }:$src1, (imm:{ *:[i64] })<<P:Predicate_i64immSExt32>>:$src2)  =>  (ADD64ri32:{ *:[i64] }:{ *:[i32] } GR64:{ *:[i64] }:$src1, (imm:{ *:[i64] })<<P:Predicate_i64immSExt32>>:$src2)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::ADD64ri32,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // src1
        GIR_CopyConstantAsSImm, /*NewInsnID*/0, /*OldInsnID*/1, // src2
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 16108,
        GIR_Done,
      // Label 77: @776
      GIM_Try, /*On fail goto*//*Label 78*/ 792, // Rule ID 16097 //
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::GR64RegClassID,
        // (add:{ *:[i64] } GR64:{ *:[i64] }:$src1, GR64:{ *:[i64] }:$src2)  =>  (ADD64rr:{ *:[i64] }:{ *:[i32] } GR64:{ *:[i64] }:$src1, GR64:{ *:[i64] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::ADD64rr,
        GIR_AddImplicitDef, /*InsnID*/0, X86::EFLAGS,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 16097,
        GIR_Done,
      // Label 78: @792
      GIM_Reject,
    // Label 71: @793
    GIM_Reject,
    // Label 39: @794
    GIM_Try, /*On fail goto*//*Label 79*/ 874,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v2s64,
      GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v2s64,
      GIM_Try, /*On fail goto*//*Label 80*/ 827, // Rule ID 1819 //
        GIM_CheckFeatures, GIFBS_HasAVX_NoVLX,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR128RegClassID,
        // (add:{ *:[v2i64] } VR128:{ *:[v2i64] }:$src1, VR128:{ *:[v2i64] }:$src2)  =>  (VPADDQrr:{ *:[v2i64] } VR128:{ *:[v2i64] }:$src1, VR128:{ *:[v2i64] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VPADDQrr,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 1819,
        GIR_Done,
      // Label 80: @827
      GIM_Try, /*On fail goto*//*Label 81*/ 850, // Rule ID 1821 //
        GIM_CheckFeatures, GIFBS_UseSSE2,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR128RegClassID,
        // (add:{ *:[v2i64] } VR128:{ *:[v2i64] }:$src1, VR128:{ *:[v2i64] }:$src2)  =>  (PADDQrr:{ *:[v2i64] } VR128:{ *:[v2i64] }:$src1, VR128:{ *:[v2i64] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::PADDQrr,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 1821,
        GIR_Done,
      // Label 81: @850
      GIM_Try, /*On fail goto*//*Label 82*/ 873, // Rule ID 3833 //
        GIM_CheckFeatures, GIFBS_HasAVX512_HasVLX,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128XRegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR128XRegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR128XRegClassID,
        // (add:{ *:[v2i64] } VR128X:{ *:[v2i64] }:$src1, VR128X:{ *:[v2i64] }:$src2)  =>  (VPADDQZ128rr:{ *:[v2i64] } VR128X:{ *:[v2i64] }:$src1, VR128X:{ *:[v2i64] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VPADDQZ128rr,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 3833,
        GIR_Done,
      // Label 82: @873
      GIM_Reject,
    // Label 79: @874
    GIM_Reject,
    // Label 40: @875
    GIM_Try, /*On fail goto*//*Label 83*/ 1077,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v4s32,
      GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v4s32,
      GIM_Try, /*On fail goto*//*Label 84*/ 946, // Rule ID 12364 //
        GIM_CheckFeatures, GIFBS_HasXOP,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128RegClassID,
        GIM_RecordInsn, /*DefineMI*/1, /*MI*/0, /*OpIdx*/1, // MIs[1]
        GIM_CheckOpcode, /*MI*/1, TargetOpcode::G_MUL,
        GIM_CheckType, /*MI*/1, /*Op*/1, /*Type*/GILLT_v4s32,
        GIM_CheckType, /*MI*/1, /*Op*/2, /*Type*/GILLT_v4s32,
        GIM_CheckRegBankForClass, /*MI*/1, /*Op*/1, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/1, /*Op*/2, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR128RegClassID,
        GIM_CheckIsSafeToFold, /*InsnID*/1,
        // (add:{ *:[v4i32] } (mul:{ *:[v4i32] } VR128:{ *:[v4i32] }:$src1, VR128:{ *:[v4i32] }:$src2), VR128:{ *:[v4i32] }:$src3)  =>  (VPMACSDDrr:{ *:[v4i32] } VR128:{ *:[v4i32] }:$src1, VR128:{ *:[v4i32] }:$src2, VR128:{ *:[v4i32] }:$src3)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::VPMACSDDrr,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/1, /*OpIdx*/1, // src1
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/1, /*OpIdx*/2, // src2
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/2, // src3
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 12364,
        GIR_Done,
      // Label 84: @946
      GIM_Try, /*On fail goto*//*Label 85*/ 1007, // Rule ID 18213 //
        GIM_CheckFeatures, GIFBS_HasXOP,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR128RegClassID,
        GIM_RecordInsn, /*DefineMI*/1, /*MI*/0, /*OpIdx*/2, // MIs[1]
        GIM_CheckOpcode, /*MI*/1, TargetOpcode::G_MUL,
        GIM_CheckType, /*MI*/1, /*Op*/1, /*Type*/GILLT_v4s32,
        GIM_CheckType, /*MI*/1, /*Op*/2, /*Type*/GILLT_v4s32,
        GIM_CheckRegBankForClass, /*MI*/1, /*Op*/1, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/1, /*Op*/2, /*RC*/X86::VR128RegClassID,
        GIM_CheckIsSafeToFold, /*InsnID*/1,
        // (add:{ *:[v4i32] } VR128:{ *:[v4i32] }:$src3, (mul:{ *:[v4i32] } VR128:{ *:[v4i32] }:$src1, VR128:{ *:[v4i32] }:$src2))  =>  (VPMACSDDrr:{ *:[v4i32] } VR128:{ *:[v4i32] }:$src1, VR128:{ *:[v4i32] }:$src2, VR128:{ *:[v4i32] }:$src3)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::VPMACSDDrr,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/1, /*OpIdx*/1, // src1
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/1, /*OpIdx*/2, // src2
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // src3
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 18213,
        GIR_Done,
      // Label 85: @1007
      GIM_Try, /*On fail goto*//*Label 86*/ 1030, // Rule ID 1813 //
        GIM_CheckFeatures, GIFBS_HasAVX_NoVLX,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR128RegClassID,
        // (add:{ *:[v4i32] } VR128:{ *:[v4i32] }:$src1, VR128:{ *:[v4i32] }:$src2)  =>  (VPADDDrr:{ *:[v4i32] } VR128:{ *:[v4i32] }:$src1, VR128:{ *:[v4i32] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VPADDDrr,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 1813,
        GIR_Done,
      // Label 86: @1030
      GIM_Try, /*On fail goto*//*Label 87*/ 1053, // Rule ID 1815 //
        GIM_CheckFeatures, GIFBS_UseSSE2,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR128RegClassID,
        // (add:{ *:[v4i32] } VR128:{ *:[v4i32] }:$src1, VR128:{ *:[v4i32] }:$src2)  =>  (PADDDrr:{ *:[v4i32] } VR128:{ *:[v4i32] }:$src1, VR128:{ *:[v4i32] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::PADDDrr,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 1815,
        GIR_Done,
      // Label 87: @1053
      GIM_Try, /*On fail goto*//*Label 88*/ 1076, // Rule ID 3860 //
        GIM_CheckFeatures, GIFBS_HasAVX512_HasVLX,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128XRegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR128XRegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR128XRegClassID,
        // (add:{ *:[v4i32] } VR128X:{ *:[v4i32] }:$src1, VR128X:{ *:[v4i32] }:$src2)  =>  (VPADDDZ128rr:{ *:[v4i32] } VR128X:{ *:[v4i32] }:$src1, VR128X:{ *:[v4i32] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VPADDDZ128rr,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 3860,
        GIR_Done,
      // Label 88: @1076
      GIM_Reject,
    // Label 83: @1077
    GIM_Reject,
    // Label 41: @1078
    GIM_Try, /*On fail goto*//*Label 89*/ 1135,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v4s64,
      GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v4s64,
      GIM_Try, /*On fail goto*//*Label 90*/ 1111, // Rule ID 1823 //
        GIM_CheckFeatures, GIFBS_HasAVX2_NoVLX,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR256RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR256RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR256RegClassID,
        // (add:{ *:[v4i64] } VR256:{ *:[v4i64] }:$src1, VR256:{ *:[v4i64] }:$src2)  =>  (VPADDQYrr:{ *:[v4i64] } VR256:{ *:[v4i64] }:$src1, VR256:{ *:[v4i64] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VPADDQYrr,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 1823,
        GIR_Done,
      // Label 90: @1111
      GIM_Try, /*On fail goto*//*Label 91*/ 1134, // Rule ID 3824 //
        GIM_CheckFeatures, GIFBS_HasAVX512_HasVLX,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR256XRegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR256XRegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR256XRegClassID,
        // (add:{ *:[v4i64] } VR256X:{ *:[v4i64] }:$src1, VR256X:{ *:[v4i64] }:$src2)  =>  (VPADDQZ256rr:{ *:[v4i64] } VR256X:{ *:[v4i64] }:$src1, VR256X:{ *:[v4i64] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VPADDQZ256rr,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 3824,
        GIR_Done,
      // Label 91: @1134
      GIM_Reject,
    // Label 89: @1135
    GIM_Reject,
    // Label 42: @1136
    GIM_Try, /*On fail goto*//*Label 92*/ 1338,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v8s16,
      GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v8s16,
      GIM_Try, /*On fail goto*//*Label 93*/ 1207, // Rule ID 12363 //
        GIM_CheckFeatures, GIFBS_HasXOP,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128RegClassID,
        GIM_RecordInsn, /*DefineMI*/1, /*MI*/0, /*OpIdx*/1, // MIs[1]
        GIM_CheckOpcode, /*MI*/1, TargetOpcode::G_MUL,
        GIM_CheckType, /*MI*/1, /*Op*/1, /*Type*/GILLT_v8s16,
        GIM_CheckType, /*MI*/1, /*Op*/2, /*Type*/GILLT_v8s16,
        GIM_CheckRegBankForClass, /*MI*/1, /*Op*/1, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/1, /*Op*/2, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR128RegClassID,
        GIM_CheckIsSafeToFold, /*InsnID*/1,
        // (add:{ *:[v8i16] } (mul:{ *:[v8i16] } VR128:{ *:[v8i16] }:$src1, VR128:{ *:[v8i16] }:$src2), VR128:{ *:[v8i16] }:$src3)  =>  (VPMACSWWrr:{ *:[v8i16] } VR128:{ *:[v8i16] }:$src1, VR128:{ *:[v8i16] }:$src2, VR128:{ *:[v8i16] }:$src3)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::VPMACSWWrr,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/1, /*OpIdx*/1, // src1
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/1, /*OpIdx*/2, // src2
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/2, // src3
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 12363,
        GIR_Done,
      // Label 93: @1207
      GIM_Try, /*On fail goto*//*Label 94*/ 1268, // Rule ID 18212 //
        GIM_CheckFeatures, GIFBS_HasXOP,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR128RegClassID,
        GIM_RecordInsn, /*DefineMI*/1, /*MI*/0, /*OpIdx*/2, // MIs[1]
        GIM_CheckOpcode, /*MI*/1, TargetOpcode::G_MUL,
        GIM_CheckType, /*MI*/1, /*Op*/1, /*Type*/GILLT_v8s16,
        GIM_CheckType, /*MI*/1, /*Op*/2, /*Type*/GILLT_v8s16,
        GIM_CheckRegBankForClass, /*MI*/1, /*Op*/1, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/1, /*Op*/2, /*RC*/X86::VR128RegClassID,
        GIM_CheckIsSafeToFold, /*InsnID*/1,
        // (add:{ *:[v8i16] } VR128:{ *:[v8i16] }:$src3, (mul:{ *:[v8i16] } VR128:{ *:[v8i16] }:$src1, VR128:{ *:[v8i16] }:$src2))  =>  (VPMACSWWrr:{ *:[v8i16] } VR128:{ *:[v8i16] }:$src1, VR128:{ *:[v8i16] }:$src2, VR128:{ *:[v8i16] }:$src3)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::VPMACSWWrr,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/1, /*OpIdx*/1, // src1
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/1, /*OpIdx*/2, // src2
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // src3
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 18212,
        GIR_Done,
      // Label 94: @1268
      GIM_Try, /*On fail goto*//*Label 95*/ 1291, // Rule ID 1807 //
        GIM_CheckFeatures, GIFBS_HasAVX_NoVLX_Or_NoBWI,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR128RegClassID,
        // (add:{ *:[v8i16] } VR128:{ *:[v8i16] }:$src1, VR128:{ *:[v8i16] }:$src2)  =>  (VPADDWrr:{ *:[v8i16] } VR128:{ *:[v8i16] }:$src1, VR128:{ *:[v8i16] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VPADDWrr,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 1807,
        GIR_Done,
      // Label 95: @1291
      GIM_Try, /*On fail goto*//*Label 96*/ 1314, // Rule ID 1809 //
        GIM_CheckFeatures, GIFBS_UseSSE2,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR128RegClassID,
        // (add:{ *:[v8i16] } VR128:{ *:[v8i16] }:$src1, VR128:{ *:[v8i16] }:$src2)  =>  (PADDWrr:{ *:[v8i16] } VR128:{ *:[v8i16] }:$src1, VR128:{ *:[v8i16] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::PADDWrr,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 1809,
        GIR_Done,
      // Label 96: @1314
      GIM_Try, /*On fail goto*//*Label 97*/ 1337, // Rule ID 3881 //
        GIM_CheckFeatures, GIFBS_HasBWI_HasVLX,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128XRegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR128XRegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR128XRegClassID,
        // (add:{ *:[v8i16] } VR128X:{ *:[v8i16] }:$src1, VR128X:{ *:[v8i16] }:$src2)  =>  (VPADDWZ128rr:{ *:[v8i16] } VR128X:{ *:[v8i16] }:$src1, VR128X:{ *:[v8i16] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VPADDWZ128rr,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 3881,
        GIR_Done,
      // Label 97: @1337
      GIM_Reject,
    // Label 92: @1338
    GIM_Reject,
    // Label 43: @1339
    GIM_Try, /*On fail goto*//*Label 98*/ 1396,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v8s32,
      GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v8s32,
      GIM_Try, /*On fail goto*//*Label 99*/ 1372, // Rule ID 1817 //
        GIM_CheckFeatures, GIFBS_HasAVX2_NoVLX,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR256RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR256RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR256RegClassID,
        // (add:{ *:[v8i32] } VR256:{ *:[v8i32] }:$src1, VR256:{ *:[v8i32] }:$src2)  =>  (VPADDDYrr:{ *:[v8i32] } VR256:{ *:[v8i32] }:$src1, VR256:{ *:[v8i32] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VPADDDYrr,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 1817,
        GIR_Done,
      // Label 99: @1372
      GIM_Try, /*On fail goto*//*Label 100*/ 1395, // Rule ID 3851 //
        GIM_CheckFeatures, GIFBS_HasAVX512_HasVLX,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR256XRegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR256XRegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR256XRegClassID,
        // (add:{ *:[v8i32] } VR256X:{ *:[v8i32] }:$src1, VR256X:{ *:[v8i32] }:$src2)  =>  (VPADDDZ256rr:{ *:[v8i32] } VR256X:{ *:[v8i32] }:$src1, VR256X:{ *:[v8i32] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VPADDDZ256rr,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 3851,
        GIR_Done,
      // Label 100: @1395
      GIM_Reject,
    // Label 98: @1396
    GIM_Reject,
    // Label 44: @1397
    GIM_Try, /*On fail goto*//*Label 101*/ 1428, // Rule ID 3815 //
      GIM_CheckFeatures, GIFBS_HasAVX512,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v8s64,
      GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v8s64,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR512RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR512RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR512RegClassID,
      // (add:{ *:[v8i64] } VR512:{ *:[v8i64] }:$src1, VR512:{ *:[v8i64] }:$src2)  =>  (VPADDQZrr:{ *:[v8i64] } VR512:{ *:[v8i64] }:$src1, VR512:{ *:[v8i64] }:$src2)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VPADDQZrr,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 3815,
      GIR_Done,
    // Label 101: @1428
    GIM_Reject,
    // Label 45: @1429
    GIM_Try, /*On fail goto*//*Label 102*/ 1509,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v16s8,
      GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v16s8,
      GIM_Try, /*On fail goto*//*Label 103*/ 1462, // Rule ID 1801 //
        GIM_CheckFeatures, GIFBS_HasAVX_NoVLX_Or_NoBWI,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR128RegClassID,
        // (add:{ *:[v16i8] } VR128:{ *:[v16i8] }:$src1, VR128:{ *:[v16i8] }:$src2)  =>  (VPADDBrr:{ *:[v16i8] } VR128:{ *:[v16i8] }:$src1, VR128:{ *:[v16i8] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VPADDBrr,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 1801,
        GIR_Done,
      // Label 103: @1462
      GIM_Try, /*On fail goto*//*Label 104*/ 1485, // Rule ID 1803 //
        GIM_CheckFeatures, GIFBS_UseSSE2,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR128RegClassID,
        // (add:{ *:[v16i8] } VR128:{ *:[v16i8] }:$src1, VR128:{ *:[v16i8] }:$src2)  =>  (PADDBrr:{ *:[v16i8] } VR128:{ *:[v16i8] }:$src1, VR128:{ *:[v16i8] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::PADDBrr,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 1803,
        GIR_Done,
      // Label 104: @1485
      GIM_Try, /*On fail goto*//*Label 105*/ 1508, // Rule ID 3899 //
        GIM_CheckFeatures, GIFBS_HasBWI_HasVLX,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128XRegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR128XRegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR128XRegClassID,
        // (add:{ *:[v16i8] } VR128X:{ *:[v16i8] }:$src1, VR128X:{ *:[v16i8] }:$src2)  =>  (VPADDBZ128rr:{ *:[v16i8] } VR128X:{ *:[v16i8] }:$src1, VR128X:{ *:[v16i8] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VPADDBZ128rr,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 3899,
        GIR_Done,
      // Label 105: @1508
      GIM_Reject,
    // Label 102: @1509
    GIM_Reject,
    // Label 46: @1510
    GIM_Try, /*On fail goto*//*Label 106*/ 1567,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v16s16,
      GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v16s16,
      GIM_Try, /*On fail goto*//*Label 107*/ 1543, // Rule ID 1811 //
        GIM_CheckFeatures, GIFBS_HasAVX2_NoVLX_Or_NoBWI,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR256RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR256RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR256RegClassID,
        // (add:{ *:[v16i16] } VR256:{ *:[v16i16] }:$src1, VR256:{ *:[v16i16] }:$src2)  =>  (VPADDWYrr:{ *:[v16i16] } VR256:{ *:[v16i16] }:$src1, VR256:{ *:[v16i16] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VPADDWYrr,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 1811,
        GIR_Done,
      // Label 107: @1543
      GIM_Try, /*On fail goto*//*Label 108*/ 1566, // Rule ID 3875 //
        GIM_CheckFeatures, GIFBS_HasBWI_HasVLX,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR256XRegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR256XRegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR256XRegClassID,
        // (add:{ *:[v16i16] } VR256X:{ *:[v16i16] }:$src1, VR256X:{ *:[v16i16] }:$src2)  =>  (VPADDWZ256rr:{ *:[v16i16] } VR256X:{ *:[v16i16] }:$src1, VR256X:{ *:[v16i16] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VPADDWZ256rr,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 3875,
        GIR_Done,
      // Label 108: @1566
      GIM_Reject,
    // Label 106: @1567
    GIM_Reject,
    // Label 47: @1568
    GIM_Try, /*On fail goto*//*Label 109*/ 1599, // Rule ID 3842 //
      GIM_CheckFeatures, GIFBS_HasAVX512,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v16s32,
      GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v16s32,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR512RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR512RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR512RegClassID,
      // (add:{ *:[v16i32] } VR512:{ *:[v16i32] }:$src1, VR512:{ *:[v16i32] }:$src2)  =>  (VPADDDZrr:{ *:[v16i32] } VR512:{ *:[v16i32] }:$src1, VR512:{ *:[v16i32] }:$src2)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VPADDDZrr,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 3842,
      GIR_Done,
    // Label 109: @1599
    GIM_Reject,
    // Label 48: @1600
    GIM_Try, /*On fail goto*//*Label 110*/ 1657,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v32s8,
      GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v32s8,
      GIM_Try, /*On fail goto*//*Label 111*/ 1633, // Rule ID 1805 //
        GIM_CheckFeatures, GIFBS_HasAVX2_NoVLX_Or_NoBWI,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR256RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR256RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR256RegClassID,
        // (add:{ *:[v32i8] } VR256:{ *:[v32i8] }:$src1, VR256:{ *:[v32i8] }:$src2)  =>  (VPADDBYrr:{ *:[v32i8] } VR256:{ *:[v32i8] }:$src1, VR256:{ *:[v32i8] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VPADDBYrr,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 1805,
        GIR_Done,
      // Label 111: @1633
      GIM_Try, /*On fail goto*//*Label 112*/ 1656, // Rule ID 3893 //
        GIM_CheckFeatures, GIFBS_HasBWI_HasVLX,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR256XRegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR256XRegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR256XRegClassID,
        // (add:{ *:[v32i8] } VR256X:{ *:[v32i8] }:$src1, VR256X:{ *:[v32i8] }:$src2)  =>  (VPADDBZ256rr:{ *:[v32i8] } VR256X:{ *:[v32i8] }:$src1, VR256X:{ *:[v32i8] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VPADDBZ256rr,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 3893,
        GIR_Done,
      // Label 112: @1656
      GIM_Reject,
    // Label 110: @1657
    GIM_Reject,
    // Label 49: @1658
    GIM_Try, /*On fail goto*//*Label 113*/ 1689, // Rule ID 3869 //
      GIM_CheckFeatures, GIFBS_HasBWI,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v32s16,
      GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v32s16,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR512RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR512RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR512RegClassID,
      // (add:{ *:[v32i16] } VR512:{ *:[v32i16] }:$src1, VR512:{ *:[v32i16] }:$src2)  =>  (VPADDWZrr:{ *:[v32i16] } VR512:{ *:[v32i16] }:$src1, VR512:{ *:[v32i16] }:$src2)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VPADDWZrr,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 3869,
      GIR_Done,
    // Label 113: @1689
    GIM_Reject,
    // Label 50: @1690
    GIM_Try, /*On fail goto*//*Label 114*/ 1721, // Rule ID 3887 //
      GIM_CheckFeatures, GIFBS_HasBWI,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v64s8,
      GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v64s8,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR512RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR512RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR512RegClassID,
      // (add:{ *:[v64i8] } VR512:{ *:[v64i8] }:$src1, VR512:{ *:[v64i8] }:$src2)  =>  (VPADDBZrr:{ *:[v64i8] } VR512:{ *:[v64i8] }:$src1, VR512:{ *:[v64i8] }:$src2)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VPADDBZrr,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 3887,
      GIR_Done,
    // Label 114: @1721
    GIM_Reject,
    // Label 51: @1722
    GIM_Reject,
    // Label 1: @1723
    GIM_SwitchType, /*MI*/0, /*Op*/0, /*[*/1, 25, /*)*//*default:*//*Label 131*/ 2803,
    /*GILLT_s8*//*Label 115*/ 1753,
    /*GILLT_s16*//*Label 116*/ 1819,
    /*GILLT_s32*//*Label 117*/ 1918,
    /*GILLT_s64*//*Label 118*/ 2017, 0, 0, 0,
    /*GILLT_v2s64*//*Label 119*/ 2119, 0,
    /*GILLT_v4s32*//*Label 120*/ 2200,
    /*GILLT_v4s64*//*Label 121*/ 2281, 0,
    /*GILLT_v8s16*//*Label 122*/ 2339,
    /*GILLT_v8s32*//*Label 123*/ 2420,
    /*GILLT_v8s64*//*Label 124*/ 2478, 0,
    /*GILLT_v16s8*//*Label 125*/ 2510,
    /*GILLT_v16s16*//*Label 126*/ 2591,
    /*GILLT_v16s32*//*Label 127*/ 2649, 0,
    /*GILLT_v32s8*//*Label 128*/ 2681,
    /*GILLT_v32s16*//*Label 129*/ 2739, 0,
    /*GILLT_v64s8*//*Label 130*/ 2771,
    // Label 115: @1753
    GIM_Try, /*On fail goto*//*Label 132*/ 1818,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_s8,
      GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_s8,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::GR8RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::GR8RegClassID,
      GIM_Try, /*On fail goto*//*Label 133*/ 1801, // Rule ID 16117 //
        GIM_RecordInsn, /*DefineMI*/1, /*MI*/0, /*OpIdx*/2, // MIs[1]
        GIM_CheckOpcode, /*MI*/1, TargetOpcode::G_CONSTANT,
        // MIs[1] Operand 1
        // No operand predicates
        GIM_CheckIsSafeToFold, /*InsnID*/1,
        // (sub:{ *:[i8] } GR8:{ *:[i8] }:$src1, (imm:{ *:[i8] }):$src2)  =>  (SUB8ri:{ *:[i8] }:{ *:[i32] } GR8:{ *:[i8] }:$src1, (imm:{ *:[i8] }):$src2)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::SUB8ri,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // src1
        GIR_CopyConstantAsSImm, /*NewInsnID*/0, /*OldInsnID*/1, // src2
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 16117,
        GIR_Done,
      // Label 133: @1801
      GIM_Try, /*On fail goto*//*Label 134*/ 1817, // Rule ID 16109 //
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::GR8RegClassID,
        // (sub:{ *:[i8] } GR8:{ *:[i8] }:$src1, GR8:{ *:[i8] }:$src2)  =>  (SUB8rr:{ *:[i8] }:{ *:[i32] } GR8:{ *:[i8] }:$src1, GR8:{ *:[i8] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::SUB8rr,
        GIR_AddImplicitDef, /*InsnID*/0, X86::EFLAGS,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 16109,
        GIR_Done,
      // Label 134: @1817
      GIM_Reject,
    // Label 132: @1818
    GIM_Reject,
    // Label 116: @1819
    GIM_Try, /*On fail goto*//*Label 135*/ 1917,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_s16,
      GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_s16,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::GR16RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::GR16RegClassID,
      GIM_Try, /*On fail goto*//*Label 136*/ 1870, // Rule ID 16120 //
        GIM_RecordInsn, /*DefineMI*/1, /*MI*/0, /*OpIdx*/2, // MIs[1]
        GIM_CheckOpcode, /*MI*/1, TargetOpcode::G_CONSTANT,
        GIM_CheckI64ImmPredicate, /*MI*/1, /*Predicate*/GIPFP_I64_Predicate_i16immSExt8,
        // MIs[1] Operand 1
        // No operand predicates
        GIM_CheckIsSafeToFold, /*InsnID*/1,
        // (sub:{ *:[i16] } GR16:{ *:[i16] }:$src1, (imm:{ *:[i16] })<<P:Predicate_i16immSExt8>>:$src2)  =>  (SUB16ri8:{ *:[i16] }:{ *:[i32] } GR16:{ *:[i16] }:$src1, (imm:{ *:[i16] })<<P:Predicate_i16immSExt8>>:$src2)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::SUB16ri8,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // src1
        GIR_CopyConstantAsSImm, /*NewInsnID*/0, /*OldInsnID*/1, // src2
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 16120,
        GIR_Done,
      // Label 136: @1870
      GIM_Try, /*On fail goto*//*Label 137*/ 1900, // Rule ID 16118 //
        GIM_RecordInsn, /*DefineMI*/1, /*MI*/0, /*OpIdx*/2, // MIs[1]
        GIM_CheckOpcode, /*MI*/1, TargetOpcode::G_CONSTANT,
        // MIs[1] Operand 1
        // No operand predicates
        GIM_CheckIsSafeToFold, /*InsnID*/1,
        // (sub:{ *:[i16] } GR16:{ *:[i16] }:$src1, (imm:{ *:[i16] }):$src2)  =>  (SUB16ri:{ *:[i16] }:{ *:[i32] } GR16:{ *:[i16] }:$src1, (imm:{ *:[i16] }):$src2)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::SUB16ri,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // src1
        GIR_CopyConstantAsSImm, /*NewInsnID*/0, /*OldInsnID*/1, // src2
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 16118,
        GIR_Done,
      // Label 137: @1900
      GIM_Try, /*On fail goto*//*Label 138*/ 1916, // Rule ID 16110 //
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::GR16RegClassID,
        // (sub:{ *:[i16] } GR16:{ *:[i16] }:$src1, GR16:{ *:[i16] }:$src2)  =>  (SUB16rr:{ *:[i16] }:{ *:[i32] } GR16:{ *:[i16] }:$src1, GR16:{ *:[i16] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::SUB16rr,
        GIR_AddImplicitDef, /*InsnID*/0, X86::EFLAGS,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 16110,
        GIR_Done,
      // Label 138: @1916
      GIM_Reject,
    // Label 135: @1917
    GIM_Reject,
    // Label 117: @1918
    GIM_Try, /*On fail goto*//*Label 139*/ 2016,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_s32,
      GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_s32,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::GR32RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::GR32RegClassID,
      GIM_Try, /*On fail goto*//*Label 140*/ 1969, // Rule ID 16121 //
        GIM_RecordInsn, /*DefineMI*/1, /*MI*/0, /*OpIdx*/2, // MIs[1]
        GIM_CheckOpcode, /*MI*/1, TargetOpcode::G_CONSTANT,
        GIM_CheckI64ImmPredicate, /*MI*/1, /*Predicate*/GIPFP_I64_Predicate_i32immSExt8,
        // MIs[1] Operand 1
        // No operand predicates
        GIM_CheckIsSafeToFold, /*InsnID*/1,
        // (sub:{ *:[i32] } GR32:{ *:[i32] }:$src1, (imm:{ *:[i32] })<<P:Predicate_i32immSExt8>>:$src2)  =>  (SUB32ri8:{ *:[i32] }:{ *:[i32] } GR32:{ *:[i32] }:$src1, (imm:{ *:[i32] })<<P:Predicate_i32immSExt8>>:$src2)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::SUB32ri8,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // src1
        GIR_CopyConstantAsSImm, /*NewInsnID*/0, /*OldInsnID*/1, // src2
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 16121,
        GIR_Done,
      // Label 140: @1969
      GIM_Try, /*On fail goto*//*Label 141*/ 1999, // Rule ID 16119 //
        GIM_RecordInsn, /*DefineMI*/1, /*MI*/0, /*OpIdx*/2, // MIs[1]
        GIM_CheckOpcode, /*MI*/1, TargetOpcode::G_CONSTANT,
        // MIs[1] Operand 1
        // No operand predicates
        GIM_CheckIsSafeToFold, /*InsnID*/1,
        // (sub:{ *:[i32] } GR32:{ *:[i32] }:$src1, (imm:{ *:[i32] }):$src2)  =>  (SUB32ri:{ *:[i32] }:{ *:[i32] } GR32:{ *:[i32] }:$src1, (imm:{ *:[i32] }):$src2)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::SUB32ri,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // src1
        GIR_CopyConstantAsSImm, /*NewInsnID*/0, /*OldInsnID*/1, // src2
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 16119,
        GIR_Done,
      // Label 141: @1999
      GIM_Try, /*On fail goto*//*Label 142*/ 2015, // Rule ID 16111 //
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::GR32RegClassID,
        // (sub:{ *:[i32] } GR32:{ *:[i32] }:$src1, GR32:{ *:[i32] }:$src2)  =>  (SUB32rr:{ *:[i32] }:{ *:[i32] } GR32:{ *:[i32] }:$src1, GR32:{ *:[i32] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::SUB32rr,
        GIR_AddImplicitDef, /*InsnID*/0, X86::EFLAGS,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 16111,
        GIR_Done,
      // Label 142: @2015
      GIM_Reject,
    // Label 139: @2016
    GIM_Reject,
    // Label 118: @2017
    GIM_Try, /*On fail goto*//*Label 143*/ 2118,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_s64,
      GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_s64,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::GR64RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::GR64RegClassID,
      GIM_Try, /*On fail goto*//*Label 144*/ 2068, // Rule ID 16122 //
        GIM_RecordInsn, /*DefineMI*/1, /*MI*/0, /*OpIdx*/2, // MIs[1]
        GIM_CheckOpcode, /*MI*/1, TargetOpcode::G_CONSTANT,
        GIM_CheckI64ImmPredicate, /*MI*/1, /*Predicate*/GIPFP_I64_Predicate_i64immSExt8,
        // MIs[1] Operand 1
        // No operand predicates
        GIM_CheckIsSafeToFold, /*InsnID*/1,
        // (sub:{ *:[i64] } GR64:{ *:[i64] }:$src1, (imm:{ *:[i64] })<<P:Predicate_i64immSExt8>>:$src2)  =>  (SUB64ri8:{ *:[i64] }:{ *:[i32] } GR64:{ *:[i64] }:$src1, (imm:{ *:[i64] })<<P:Predicate_i64immSExt8>>:$src2)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::SUB64ri8,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // src1
        GIR_CopyConstantAsSImm, /*NewInsnID*/0, /*OldInsnID*/1, // src2
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 16122,
        GIR_Done,
      // Label 144: @2068
      GIM_Try, /*On fail goto*//*Label 145*/ 2101, // Rule ID 16123 //
        GIM_RecordInsn, /*DefineMI*/1, /*MI*/0, /*OpIdx*/2, // MIs[1]
        GIM_CheckOpcode, /*MI*/1, TargetOpcode::G_CONSTANT,
        GIM_CheckI64ImmPredicate, /*MI*/1, /*Predicate*/GIPFP_I64_Predicate_i64immSExt32,
        // MIs[1] Operand 1
        // No operand predicates
        GIM_CheckIsSafeToFold, /*InsnID*/1,
        // (sub:{ *:[i64] } GR64:{ *:[i64] }:$src1, (imm:{ *:[i64] })<<P:Predicate_i64immSExt32>>:$src2)  =>  (SUB64ri32:{ *:[i64] }:{ *:[i32] } GR64:{ *:[i64] }:$src1, (imm:{ *:[i64] })<<P:Predicate_i64immSExt32>>:$src2)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::SUB64ri32,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // src1
        GIR_CopyConstantAsSImm, /*NewInsnID*/0, /*OldInsnID*/1, // src2
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 16123,
        GIR_Done,
      // Label 145: @2101
      GIM_Try, /*On fail goto*//*Label 146*/ 2117, // Rule ID 16112 //
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::GR64RegClassID,
        // (sub:{ *:[i64] } GR64:{ *:[i64] }:$src1, GR64:{ *:[i64] }:$src2)  =>  (SUB64rr:{ *:[i64] }:{ *:[i32] } GR64:{ *:[i64] }:$src1, GR64:{ *:[i64] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::SUB64rr,
        GIR_AddImplicitDef, /*InsnID*/0, X86::EFLAGS,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 16112,
        GIR_Done,
      // Label 146: @2117
      GIM_Reject,
    // Label 143: @2118
    GIM_Reject,
    // Label 119: @2119
    GIM_Try, /*On fail goto*//*Label 147*/ 2199,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v2s64,
      GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v2s64,
      GIM_Try, /*On fail goto*//*Label 148*/ 2152, // Rule ID 1885 //
        GIM_CheckFeatures, GIFBS_HasAVX_NoVLX,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR128RegClassID,
        // (sub:{ *:[v2i64] } VR128:{ *:[v2i64] }:$src1, VR128:{ *:[v2i64] }:$src2)  =>  (VPSUBQrr:{ *:[v2i64] } VR128:{ *:[v2i64] }:$src1, VR128:{ *:[v2i64] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VPSUBQrr,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 1885,
        GIR_Done,
      // Label 148: @2152
      GIM_Try, /*On fail goto*//*Label 149*/ 2175, // Rule ID 1887 //
        GIM_CheckFeatures, GIFBS_UseSSE2,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR128RegClassID,
        // (sub:{ *:[v2i64] } VR128:{ *:[v2i64] }:$src1, VR128:{ *:[v2i64] }:$src2)  =>  (PSUBQrr:{ *:[v2i64] } VR128:{ *:[v2i64] }:$src1, VR128:{ *:[v2i64] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::PSUBQrr,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 1887,
        GIR_Done,
      // Label 149: @2175
      GIM_Try, /*On fail goto*//*Label 150*/ 2198, // Rule ID 3923 //
        GIM_CheckFeatures, GIFBS_HasAVX512_HasVLX,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128XRegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR128XRegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR128XRegClassID,
        // (sub:{ *:[v2i64] } VR128X:{ *:[v2i64] }:$src1, VR128X:{ *:[v2i64] }:$src2)  =>  (VPSUBQZ128rr:{ *:[v2i64] } VR128X:{ *:[v2i64] }:$src1, VR128X:{ *:[v2i64] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VPSUBQZ128rr,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 3923,
        GIR_Done,
      // Label 150: @2198
      GIM_Reject,
    // Label 147: @2199
    GIM_Reject,
    // Label 120: @2200
    GIM_Try, /*On fail goto*//*Label 151*/ 2280,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v4s32,
      GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v4s32,
      GIM_Try, /*On fail goto*//*Label 152*/ 2233, // Rule ID 1879 //
        GIM_CheckFeatures, GIFBS_HasAVX_NoVLX,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR128RegClassID,
        // (sub:{ *:[v4i32] } VR128:{ *:[v4i32] }:$src1, VR128:{ *:[v4i32] }:$src2)  =>  (VPSUBDrr:{ *:[v4i32] } VR128:{ *:[v4i32] }:$src1, VR128:{ *:[v4i32] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VPSUBDrr,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 1879,
        GIR_Done,
      // Label 152: @2233
      GIM_Try, /*On fail goto*//*Label 153*/ 2256, // Rule ID 1881 //
        GIM_CheckFeatures, GIFBS_UseSSE2,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR128RegClassID,
        // (sub:{ *:[v4i32] } VR128:{ *:[v4i32] }:$src1, VR128:{ *:[v4i32] }:$src2)  =>  (PSUBDrr:{ *:[v4i32] } VR128:{ *:[v4i32] }:$src1, VR128:{ *:[v4i32] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::PSUBDrr,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 1881,
        GIR_Done,
      // Label 153: @2256
      GIM_Try, /*On fail goto*//*Label 154*/ 2279, // Rule ID 3950 //
        GIM_CheckFeatures, GIFBS_HasAVX512_HasVLX,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128XRegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR128XRegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR128XRegClassID,
        // (sub:{ *:[v4i32] } VR128X:{ *:[v4i32] }:$src1, VR128X:{ *:[v4i32] }:$src2)  =>  (VPSUBDZ128rr:{ *:[v4i32] } VR128X:{ *:[v4i32] }:$src1, VR128X:{ *:[v4i32] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VPSUBDZ128rr,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 3950,
        GIR_Done,
      // Label 154: @2279
      GIM_Reject,
    // Label 151: @2280
    GIM_Reject,
    // Label 121: @2281
    GIM_Try, /*On fail goto*//*Label 155*/ 2338,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v4s64,
      GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v4s64,
      GIM_Try, /*On fail goto*//*Label 156*/ 2314, // Rule ID 1889 //
        GIM_CheckFeatures, GIFBS_HasAVX2_NoVLX,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR256RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR256RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR256RegClassID,
        // (sub:{ *:[v4i64] } VR256:{ *:[v4i64] }:$src1, VR256:{ *:[v4i64] }:$src2)  =>  (VPSUBQYrr:{ *:[v4i64] } VR256:{ *:[v4i64] }:$src1, VR256:{ *:[v4i64] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VPSUBQYrr,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 1889,
        GIR_Done,
      // Label 156: @2314
      GIM_Try, /*On fail goto*//*Label 157*/ 2337, // Rule ID 3914 //
        GIM_CheckFeatures, GIFBS_HasAVX512_HasVLX,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR256XRegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR256XRegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR256XRegClassID,
        // (sub:{ *:[v4i64] } VR256X:{ *:[v4i64] }:$src1, VR256X:{ *:[v4i64] }:$src2)  =>  (VPSUBQZ256rr:{ *:[v4i64] } VR256X:{ *:[v4i64] }:$src1, VR256X:{ *:[v4i64] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VPSUBQZ256rr,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 3914,
        GIR_Done,
      // Label 157: @2337
      GIM_Reject,
    // Label 155: @2338
    GIM_Reject,
    // Label 122: @2339
    GIM_Try, /*On fail goto*//*Label 158*/ 2419,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v8s16,
      GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v8s16,
      GIM_Try, /*On fail goto*//*Label 159*/ 2372, // Rule ID 1873 //
        GIM_CheckFeatures, GIFBS_HasAVX_NoVLX_Or_NoBWI,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR128RegClassID,
        // (sub:{ *:[v8i16] } VR128:{ *:[v8i16] }:$src1, VR128:{ *:[v8i16] }:$src2)  =>  (VPSUBWrr:{ *:[v8i16] } VR128:{ *:[v8i16] }:$src1, VR128:{ *:[v8i16] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VPSUBWrr,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 1873,
        GIR_Done,
      // Label 159: @2372
      GIM_Try, /*On fail goto*//*Label 160*/ 2395, // Rule ID 1875 //
        GIM_CheckFeatures, GIFBS_UseSSE2,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR128RegClassID,
        // (sub:{ *:[v8i16] } VR128:{ *:[v8i16] }:$src1, VR128:{ *:[v8i16] }:$src2)  =>  (PSUBWrr:{ *:[v8i16] } VR128:{ *:[v8i16] }:$src1, VR128:{ *:[v8i16] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::PSUBWrr,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 1875,
        GIR_Done,
      // Label 160: @2395
      GIM_Try, /*On fail goto*//*Label 161*/ 2418, // Rule ID 3971 //
        GIM_CheckFeatures, GIFBS_HasBWI_HasVLX,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128XRegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR128XRegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR128XRegClassID,
        // (sub:{ *:[v8i16] } VR128X:{ *:[v8i16] }:$src1, VR128X:{ *:[v8i16] }:$src2)  =>  (VPSUBWZ128rr:{ *:[v8i16] } VR128X:{ *:[v8i16] }:$src1, VR128X:{ *:[v8i16] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VPSUBWZ128rr,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 3971,
        GIR_Done,
      // Label 161: @2418
      GIM_Reject,
    // Label 158: @2419
    GIM_Reject,
    // Label 123: @2420
    GIM_Try, /*On fail goto*//*Label 162*/ 2477,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v8s32,
      GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v8s32,
      GIM_Try, /*On fail goto*//*Label 163*/ 2453, // Rule ID 1883 //
        GIM_CheckFeatures, GIFBS_HasAVX2_NoVLX,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR256RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR256RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR256RegClassID,
        // (sub:{ *:[v8i32] } VR256:{ *:[v8i32] }:$src1, VR256:{ *:[v8i32] }:$src2)  =>  (VPSUBDYrr:{ *:[v8i32] } VR256:{ *:[v8i32] }:$src1, VR256:{ *:[v8i32] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VPSUBDYrr,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 1883,
        GIR_Done,
      // Label 163: @2453
      GIM_Try, /*On fail goto*//*Label 164*/ 2476, // Rule ID 3941 //
        GIM_CheckFeatures, GIFBS_HasAVX512_HasVLX,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR256XRegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR256XRegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR256XRegClassID,
        // (sub:{ *:[v8i32] } VR256X:{ *:[v8i32] }:$src1, VR256X:{ *:[v8i32] }:$src2)  =>  (VPSUBDZ256rr:{ *:[v8i32] } VR256X:{ *:[v8i32] }:$src1, VR256X:{ *:[v8i32] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VPSUBDZ256rr,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 3941,
        GIR_Done,
      // Label 164: @2476
      GIM_Reject,
    // Label 162: @2477
    GIM_Reject,
    // Label 124: @2478
    GIM_Try, /*On fail goto*//*Label 165*/ 2509, // Rule ID 3905 //
      GIM_CheckFeatures, GIFBS_HasAVX512,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v8s64,
      GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v8s64,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR512RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR512RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR512RegClassID,
      // (sub:{ *:[v8i64] } VR512:{ *:[v8i64] }:$src1, VR512:{ *:[v8i64] }:$src2)  =>  (VPSUBQZrr:{ *:[v8i64] } VR512:{ *:[v8i64] }:$src1, VR512:{ *:[v8i64] }:$src2)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VPSUBQZrr,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 3905,
      GIR_Done,
    // Label 165: @2509
    GIM_Reject,
    // Label 125: @2510
    GIM_Try, /*On fail goto*//*Label 166*/ 2590,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v16s8,
      GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v16s8,
      GIM_Try, /*On fail goto*//*Label 167*/ 2543, // Rule ID 1867 //
        GIM_CheckFeatures, GIFBS_HasAVX_NoVLX_Or_NoBWI,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR128RegClassID,
        // (sub:{ *:[v16i8] } VR128:{ *:[v16i8] }:$src1, VR128:{ *:[v16i8] }:$src2)  =>  (VPSUBBrr:{ *:[v16i8] } VR128:{ *:[v16i8] }:$src1, VR128:{ *:[v16i8] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VPSUBBrr,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 1867,
        GIR_Done,
      // Label 167: @2543
      GIM_Try, /*On fail goto*//*Label 168*/ 2566, // Rule ID 1869 //
        GIM_CheckFeatures, GIFBS_UseSSE2,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR128RegClassID,
        // (sub:{ *:[v16i8] } VR128:{ *:[v16i8] }:$src1, VR128:{ *:[v16i8] }:$src2)  =>  (PSUBBrr:{ *:[v16i8] } VR128:{ *:[v16i8] }:$src1, VR128:{ *:[v16i8] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::PSUBBrr,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 1869,
        GIR_Done,
      // Label 168: @2566
      GIM_Try, /*On fail goto*//*Label 169*/ 2589, // Rule ID 3989 //
        GIM_CheckFeatures, GIFBS_HasBWI_HasVLX,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128XRegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR128XRegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR128XRegClassID,
        // (sub:{ *:[v16i8] } VR128X:{ *:[v16i8] }:$src1, VR128X:{ *:[v16i8] }:$src2)  =>  (VPSUBBZ128rr:{ *:[v16i8] } VR128X:{ *:[v16i8] }:$src1, VR128X:{ *:[v16i8] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VPSUBBZ128rr,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 3989,
        GIR_Done,
      // Label 169: @2589
      GIM_Reject,
    // Label 166: @2590
    GIM_Reject,
    // Label 126: @2591
    GIM_Try, /*On fail goto*//*Label 170*/ 2648,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v16s16,
      GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v16s16,
      GIM_Try, /*On fail goto*//*Label 171*/ 2624, // Rule ID 1877 //
        GIM_CheckFeatures, GIFBS_HasAVX2_NoVLX_Or_NoBWI,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR256RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR256RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR256RegClassID,
        // (sub:{ *:[v16i16] } VR256:{ *:[v16i16] }:$src1, VR256:{ *:[v16i16] }:$src2)  =>  (VPSUBWYrr:{ *:[v16i16] } VR256:{ *:[v16i16] }:$src1, VR256:{ *:[v16i16] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VPSUBWYrr,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 1877,
        GIR_Done,
      // Label 171: @2624
      GIM_Try, /*On fail goto*//*Label 172*/ 2647, // Rule ID 3965 //
        GIM_CheckFeatures, GIFBS_HasBWI_HasVLX,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR256XRegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR256XRegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR256XRegClassID,
        // (sub:{ *:[v16i16] } VR256X:{ *:[v16i16] }:$src1, VR256X:{ *:[v16i16] }:$src2)  =>  (VPSUBWZ256rr:{ *:[v16i16] } VR256X:{ *:[v16i16] }:$src1, VR256X:{ *:[v16i16] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VPSUBWZ256rr,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 3965,
        GIR_Done,
      // Label 172: @2647
      GIM_Reject,
    // Label 170: @2648
    GIM_Reject,
    // Label 127: @2649
    GIM_Try, /*On fail goto*//*Label 173*/ 2680, // Rule ID 3932 //
      GIM_CheckFeatures, GIFBS_HasAVX512,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v16s32,
      GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v16s32,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR512RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR512RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR512RegClassID,
      // (sub:{ *:[v16i32] } VR512:{ *:[v16i32] }:$src1, VR512:{ *:[v16i32] }:$src2)  =>  (VPSUBDZrr:{ *:[v16i32] } VR512:{ *:[v16i32] }:$src1, VR512:{ *:[v16i32] }:$src2)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VPSUBDZrr,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 3932,
      GIR_Done,
    // Label 173: @2680
    GIM_Reject,
    // Label 128: @2681
    GIM_Try, /*On fail goto*//*Label 174*/ 2738,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v32s8,
      GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v32s8,
      GIM_Try, /*On fail goto*//*Label 175*/ 2714, // Rule ID 1871 //
        GIM_CheckFeatures, GIFBS_HasAVX2_NoVLX_Or_NoBWI,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR256RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR256RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR256RegClassID,
        // (sub:{ *:[v32i8] } VR256:{ *:[v32i8] }:$src1, VR256:{ *:[v32i8] }:$src2)  =>  (VPSUBBYrr:{ *:[v32i8] } VR256:{ *:[v32i8] }:$src1, VR256:{ *:[v32i8] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VPSUBBYrr,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 1871,
        GIR_Done,
      // Label 175: @2714
      GIM_Try, /*On fail goto*//*Label 176*/ 2737, // Rule ID 3983 //
        GIM_CheckFeatures, GIFBS_HasBWI_HasVLX,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR256XRegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR256XRegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR256XRegClassID,
        // (sub:{ *:[v32i8] } VR256X:{ *:[v32i8] }:$src1, VR256X:{ *:[v32i8] }:$src2)  =>  (VPSUBBZ256rr:{ *:[v32i8] } VR256X:{ *:[v32i8] }:$src1, VR256X:{ *:[v32i8] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VPSUBBZ256rr,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 3983,
        GIR_Done,
      // Label 176: @2737
      GIM_Reject,
    // Label 174: @2738
    GIM_Reject,
    // Label 129: @2739
    GIM_Try, /*On fail goto*//*Label 177*/ 2770, // Rule ID 3959 //
      GIM_CheckFeatures, GIFBS_HasBWI,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v32s16,
      GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v32s16,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR512RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR512RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR512RegClassID,
      // (sub:{ *:[v32i16] } VR512:{ *:[v32i16] }:$src1, VR512:{ *:[v32i16] }:$src2)  =>  (VPSUBWZrr:{ *:[v32i16] } VR512:{ *:[v32i16] }:$src1, VR512:{ *:[v32i16] }:$src2)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VPSUBWZrr,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 3959,
      GIR_Done,
    // Label 177: @2770
    GIM_Reject,
    // Label 130: @2771
    GIM_Try, /*On fail goto*//*Label 178*/ 2802, // Rule ID 3977 //
      GIM_CheckFeatures, GIFBS_HasBWI,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v64s8,
      GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v64s8,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR512RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR512RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR512RegClassID,
      // (sub:{ *:[v64i8] } VR512:{ *:[v64i8] }:$src1, VR512:{ *:[v64i8] }:$src2)  =>  (VPSUBBZrr:{ *:[v64i8] } VR512:{ *:[v64i8] }:$src1, VR512:{ *:[v64i8] }:$src2)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VPSUBBZrr,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 3977,
      GIR_Done,
    // Label 178: @2802
    GIM_Reject,
    // Label 131: @2803
    GIM_Reject,
    // Label 2: @2804
    GIM_SwitchType, /*MI*/0, /*Op*/0, /*[*/2, 23, /*)*//*default:*//*Label 191*/ 3569,
    /*GILLT_s16*//*Label 179*/ 2831,
    /*GILLT_s32*//*Label 180*/ 2930,
    /*GILLT_s64*//*Label 181*/ 3029, 0, 0, 0,
    /*GILLT_v2s64*//*Label 182*/ 3131, 0,
    /*GILLT_v4s32*//*Label 183*/ 3163,
    /*GILLT_v4s64*//*Label 184*/ 3244, 0,
    /*GILLT_v8s16*//*Label 185*/ 3276,
    /*GILLT_v8s32*//*Label 186*/ 3357,
    /*GILLT_v8s64*//*Label 187*/ 3415, 0, 0,
    /*GILLT_v16s16*//*Label 188*/ 3447,
    /*GILLT_v16s32*//*Label 189*/ 3505, 0, 0,
    /*GILLT_v32s16*//*Label 190*/ 3537,
    // Label 179: @2831
    GIM_Try, /*On fail goto*//*Label 192*/ 2929,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_s16,
      GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_s16,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::GR16RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::GR16RegClassID,
      GIM_Try, /*On fail goto*//*Label 193*/ 2882, // Rule ID 16138 //
        GIM_RecordInsn, /*DefineMI*/1, /*MI*/0, /*OpIdx*/2, // MIs[1]
        GIM_CheckOpcode, /*MI*/1, TargetOpcode::G_CONSTANT,
        GIM_CheckI64ImmPredicate, /*MI*/1, /*Predicate*/GIPFP_I64_Predicate_i16immSExt8,
        // MIs[1] Operand 1
        // No operand predicates
        GIM_CheckIsSafeToFold, /*InsnID*/1,
        // (mul:{ *:[i16] } GR16:{ *:[i16] }:$src1, (imm:{ *:[i16] })<<P:Predicate_i16immSExt8>>:$src2)  =>  (IMUL16rri8:{ *:[i16] }:{ *:[i32] } GR16:{ *:[i16] }:$src1, (imm:{ *:[i16] })<<P:Predicate_i16immSExt8>>:$src2)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::IMUL16rri8,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // src1
        GIR_CopyConstantAsSImm, /*NewInsnID*/0, /*OldInsnID*/1, // src2
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 16138,
        GIR_Done,
      // Label 193: @2882
      GIM_Try, /*On fail goto*//*Label 194*/ 2912, // Rule ID 16136 //
        GIM_RecordInsn, /*DefineMI*/1, /*MI*/0, /*OpIdx*/2, // MIs[1]
        GIM_CheckOpcode, /*MI*/1, TargetOpcode::G_CONSTANT,
        // MIs[1] Operand 1
        // No operand predicates
        GIM_CheckIsSafeToFold, /*InsnID*/1,
        // (mul:{ *:[i16] } GR16:{ *:[i16] }:$src1, (imm:{ *:[i16] }):$src2)  =>  (IMUL16rri:{ *:[i16] }:{ *:[i32] } GR16:{ *:[i16] }:$src1, (imm:{ *:[i16] }):$src2)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::IMUL16rri,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // src1
        GIR_CopyConstantAsSImm, /*NewInsnID*/0, /*OldInsnID*/1, // src2
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 16136,
        GIR_Done,
      // Label 194: @2912
      GIM_Try, /*On fail goto*//*Label 195*/ 2928, // Rule ID 16130 //
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::GR16RegClassID,
        // (mul:{ *:[i16] } GR16:{ *:[i16] }:$src1, GR16:{ *:[i16] }:$src2)  =>  (IMUL16rr:{ *:[i16] }:{ *:[i32] } GR16:{ *:[i16] }:$src1, GR16:{ *:[i16] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::IMUL16rr,
        GIR_AddImplicitDef, /*InsnID*/0, X86::EFLAGS,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 16130,
        GIR_Done,
      // Label 195: @2928
      GIM_Reject,
    // Label 192: @2929
    GIM_Reject,
    // Label 180: @2930
    GIM_Try, /*On fail goto*//*Label 196*/ 3028,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_s32,
      GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_s32,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::GR32RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::GR32RegClassID,
      GIM_Try, /*On fail goto*//*Label 197*/ 2981, // Rule ID 16139 //
        GIM_RecordInsn, /*DefineMI*/1, /*MI*/0, /*OpIdx*/2, // MIs[1]
        GIM_CheckOpcode, /*MI*/1, TargetOpcode::G_CONSTANT,
        GIM_CheckI64ImmPredicate, /*MI*/1, /*Predicate*/GIPFP_I64_Predicate_i32immSExt8,
        // MIs[1] Operand 1
        // No operand predicates
        GIM_CheckIsSafeToFold, /*InsnID*/1,
        // (mul:{ *:[i32] } GR32:{ *:[i32] }:$src1, (imm:{ *:[i32] })<<P:Predicate_i32immSExt8>>:$src2)  =>  (IMUL32rri8:{ *:[i32] }:{ *:[i32] } GR32:{ *:[i32] }:$src1, (imm:{ *:[i32] })<<P:Predicate_i32immSExt8>>:$src2)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::IMUL32rri8,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // src1
        GIR_CopyConstantAsSImm, /*NewInsnID*/0, /*OldInsnID*/1, // src2
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 16139,
        GIR_Done,
      // Label 197: @2981
      GIM_Try, /*On fail goto*//*Label 198*/ 3011, // Rule ID 16137 //
        GIM_RecordInsn, /*DefineMI*/1, /*MI*/0, /*OpIdx*/2, // MIs[1]
        GIM_CheckOpcode, /*MI*/1, TargetOpcode::G_CONSTANT,
        // MIs[1] Operand 1
        // No operand predicates
        GIM_CheckIsSafeToFold, /*InsnID*/1,
        // (mul:{ *:[i32] } GR32:{ *:[i32] }:$src1, (imm:{ *:[i32] }):$src2)  =>  (IMUL32rri:{ *:[i32] }:{ *:[i32] } GR32:{ *:[i32] }:$src1, (imm:{ *:[i32] }):$src2)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::IMUL32rri,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // src1
        GIR_CopyConstantAsSImm, /*NewInsnID*/0, /*OldInsnID*/1, // src2
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 16137,
        GIR_Done,
      // Label 198: @3011
      GIM_Try, /*On fail goto*//*Label 199*/ 3027, // Rule ID 16131 //
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::GR32RegClassID,
        // (mul:{ *:[i32] } GR32:{ *:[i32] }:$src1, GR32:{ *:[i32] }:$src2)  =>  (IMUL32rr:{ *:[i32] }:{ *:[i32] } GR32:{ *:[i32] }:$src1, GR32:{ *:[i32] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::IMUL32rr,
        GIR_AddImplicitDef, /*InsnID*/0, X86::EFLAGS,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 16131,
        GIR_Done,
      // Label 199: @3027
      GIM_Reject,
    // Label 196: @3028
    GIM_Reject,
    // Label 181: @3029
    GIM_Try, /*On fail goto*//*Label 200*/ 3130,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_s64,
      GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_s64,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::GR64RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::GR64RegClassID,
      GIM_Try, /*On fail goto*//*Label 201*/ 3080, // Rule ID 16140 //
        GIM_RecordInsn, /*DefineMI*/1, /*MI*/0, /*OpIdx*/2, // MIs[1]
        GIM_CheckOpcode, /*MI*/1, TargetOpcode::G_CONSTANT,
        GIM_CheckI64ImmPredicate, /*MI*/1, /*Predicate*/GIPFP_I64_Predicate_i64immSExt8,
        // MIs[1] Operand 1
        // No operand predicates
        GIM_CheckIsSafeToFold, /*InsnID*/1,
        // (mul:{ *:[i64] } GR64:{ *:[i64] }:$src1, (imm:{ *:[i64] })<<P:Predicate_i64immSExt8>>:$src2)  =>  (IMUL64rri8:{ *:[i64] }:{ *:[i32] } GR64:{ *:[i64] }:$src1, (imm:{ *:[i64] })<<P:Predicate_i64immSExt8>>:$src2)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::IMUL64rri8,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // src1
        GIR_CopyConstantAsSImm, /*NewInsnID*/0, /*OldInsnID*/1, // src2
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 16140,
        GIR_Done,
      // Label 201: @3080
      GIM_Try, /*On fail goto*//*Label 202*/ 3113, // Rule ID 16141 //
        GIM_RecordInsn, /*DefineMI*/1, /*MI*/0, /*OpIdx*/2, // MIs[1]
        GIM_CheckOpcode, /*MI*/1, TargetOpcode::G_CONSTANT,
        GIM_CheckI64ImmPredicate, /*MI*/1, /*Predicate*/GIPFP_I64_Predicate_i64immSExt32,
        // MIs[1] Operand 1
        // No operand predicates
        GIM_CheckIsSafeToFold, /*InsnID*/1,
        // (mul:{ *:[i64] } GR64:{ *:[i64] }:$src1, (imm:{ *:[i64] })<<P:Predicate_i64immSExt32>>:$src2)  =>  (IMUL64rri32:{ *:[i64] }:{ *:[i32] } GR64:{ *:[i64] }:$src1, (imm:{ *:[i64] })<<P:Predicate_i64immSExt32>>:$src2)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::IMUL64rri32,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // src1
        GIR_CopyConstantAsSImm, /*NewInsnID*/0, /*OldInsnID*/1, // src2
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 16141,
        GIR_Done,
      // Label 202: @3113
      GIM_Try, /*On fail goto*//*Label 203*/ 3129, // Rule ID 16132 //
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::GR64RegClassID,
        // (mul:{ *:[i64] } GR64:{ *:[i64] }:$src1, GR64:{ *:[i64] }:$src2)  =>  (IMUL64rr:{ *:[i64] }:{ *:[i32] } GR64:{ *:[i64] }:$src1, GR64:{ *:[i64] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::IMUL64rr,
        GIR_AddImplicitDef, /*InsnID*/0, X86::EFLAGS,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 16132,
        GIR_Done,
      // Label 203: @3129
      GIM_Reject,
    // Label 200: @3130
    GIM_Reject,
    // Label 182: @3131
    GIM_Try, /*On fail goto*//*Label 204*/ 3162, // Rule ID 4202 //
      GIM_CheckFeatures, GIFBS_HasDQI_HasVLX,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v2s64,
      GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v2s64,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128XRegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR128XRegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR128XRegClassID,
      // (mul:{ *:[v2i64] } VR128X:{ *:[v2i64] }:$src1, VR128X:{ *:[v2i64] }:$src2)  =>  (VPMULLQZ128rr:{ *:[v2i64] } VR128X:{ *:[v2i64] }:$src1, VR128X:{ *:[v2i64] }:$src2)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VPMULLQZ128rr,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 4202,
      GIR_Done,
    // Label 204: @3162
    GIM_Reject,
    // Label 183: @3163
    GIM_Try, /*On fail goto*//*Label 205*/ 3243,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v4s32,
      GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v4s32,
      GIM_Try, /*On fail goto*//*Label 206*/ 3196, // Rule ID 2507 //
        GIM_CheckFeatures, GIFBS_HasAVX_NoVLX,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR128RegClassID,
        // (mul:{ *:[v4i32] } VR128:{ *:[v4i32] }:$src1, VR128:{ *:[v4i32] }:$src2)  =>  (VPMULLDrr:{ *:[v4i32] } VR128:{ *:[v4i32] }:$src1, VR128:{ *:[v4i32] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VPMULLDrr,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 2507,
        GIR_Done,
      // Label 206: @3196
      GIM_Try, /*On fail goto*//*Label 207*/ 3219, // Rule ID 2515 //
        GIM_CheckFeatures, GIFBS_UseSSE41,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR128RegClassID,
        // (mul:{ *:[v4i32] } VR128:{ *:[v4i32] }:$src1, VR128:{ *:[v4i32] }:$src2)  =>  (PMULLDrr:{ *:[v4i32] } VR128:{ *:[v4i32] }:$src1, VR128:{ *:[v4i32] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::PMULLDrr,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 2515,
        GIR_Done,
      // Label 207: @3219
      GIM_Try, /*On fail goto*//*Label 208*/ 3242, // Rule ID 4157 //
        GIM_CheckFeatures, GIFBS_HasAVX512_HasVLX,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128XRegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR128XRegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR128XRegClassID,
        // (mul:{ *:[v4i32] } VR128X:{ *:[v4i32] }:$src1, VR128X:{ *:[v4i32] }:$src2)  =>  (VPMULLDZ128rr:{ *:[v4i32] } VR128X:{ *:[v4i32] }:$src1, VR128X:{ *:[v4i32] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VPMULLDZ128rr,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 4157,
        GIR_Done,
      // Label 208: @3242
      GIM_Reject,
    // Label 205: @3243
    GIM_Reject,
    // Label 184: @3244
    GIM_Try, /*On fail goto*//*Label 209*/ 3275, // Rule ID 4193 //
      GIM_CheckFeatures, GIFBS_HasDQI_HasVLX,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v4s64,
      GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v4s64,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR256XRegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR256XRegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR256XRegClassID,
      // (mul:{ *:[v4i64] } VR256X:{ *:[v4i64] }:$src1, VR256X:{ *:[v4i64] }:$src2)  =>  (VPMULLQZ256rr:{ *:[v4i64] } VR256X:{ *:[v4i64] }:$src1, VR256X:{ *:[v4i64] }:$src2)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VPMULLQZ256rr,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 4193,
      GIR_Done,
    // Label 209: @3275
    GIM_Reject,
    // Label 185: @3276
    GIM_Try, /*On fail goto*//*Label 210*/ 3356,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v8s16,
      GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v8s16,
      GIM_Try, /*On fail goto*//*Label 211*/ 3309, // Rule ID 1849 //
        GIM_CheckFeatures, GIFBS_HasAVX_NoVLX_Or_NoBWI,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR128RegClassID,
        // (mul:{ *:[v8i16] } VR128:{ *:[v8i16] }:$src1, VR128:{ *:[v8i16] }:$src2)  =>  (VPMULLWrr:{ *:[v8i16] } VR128:{ *:[v8i16] }:$src1, VR128:{ *:[v8i16] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VPMULLWrr,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 1849,
        GIR_Done,
      // Label 211: @3309
      GIM_Try, /*On fail goto*//*Label 212*/ 3332, // Rule ID 1851 //
        GIM_CheckFeatures, GIFBS_UseSSE2,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR128RegClassID,
        // (mul:{ *:[v8i16] } VR128:{ *:[v8i16] }:$src1, VR128:{ *:[v8i16] }:$src2)  =>  (PMULLWrr:{ *:[v8i16] } VR128:{ *:[v8i16] }:$src1, VR128:{ *:[v8i16] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::PMULLWrr,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 1851,
        GIR_Done,
      // Label 212: @3332
      GIM_Try, /*On fail goto*//*Label 213*/ 3355, // Rule ID 4178 //
        GIM_CheckFeatures, GIFBS_HasBWI_HasVLX,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128XRegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR128XRegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR128XRegClassID,
        // (mul:{ *:[v8i16] } VR128X:{ *:[v8i16] }:$src1, VR128X:{ *:[v8i16] }:$src2)  =>  (VPMULLWZ128rr:{ *:[v8i16] } VR128X:{ *:[v8i16] }:$src1, VR128X:{ *:[v8i16] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VPMULLWZ128rr,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 4178,
        GIR_Done,
      // Label 213: @3355
      GIM_Reject,
    // Label 210: @3356
    GIM_Reject,
    // Label 186: @3357
    GIM_Try, /*On fail goto*//*Label 214*/ 3414,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v8s32,
      GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v8s32,
      GIM_Try, /*On fail goto*//*Label 215*/ 3390, // Rule ID 2511 //
        GIM_CheckFeatures, GIFBS_HasAVX2_NoVLX,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR256RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR256RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR256RegClassID,
        // (mul:{ *:[v8i32] } VR256:{ *:[v8i32] }:$src1, VR256:{ *:[v8i32] }:$src2)  =>  (VPMULLDYrr:{ *:[v8i32] } VR256:{ *:[v8i32] }:$src1, VR256:{ *:[v8i32] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VPMULLDYrr,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 2511,
        GIR_Done,
      // Label 215: @3390
      GIM_Try, /*On fail goto*//*Label 216*/ 3413, // Rule ID 4148 //
        GIM_CheckFeatures, GIFBS_HasAVX512_HasVLX,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR256XRegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR256XRegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR256XRegClassID,
        // (mul:{ *:[v8i32] } VR256X:{ *:[v8i32] }:$src1, VR256X:{ *:[v8i32] }:$src2)  =>  (VPMULLDZ256rr:{ *:[v8i32] } VR256X:{ *:[v8i32] }:$src1, VR256X:{ *:[v8i32] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VPMULLDZ256rr,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 4148,
        GIR_Done,
      // Label 216: @3413
      GIM_Reject,
    // Label 214: @3414
    GIM_Reject,
    // Label 187: @3415
    GIM_Try, /*On fail goto*//*Label 217*/ 3446, // Rule ID 4184 //
      GIM_CheckFeatures, GIFBS_HasDQI,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v8s64,
      GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v8s64,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR512RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR512RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR512RegClassID,
      // (mul:{ *:[v8i64] } VR512:{ *:[v8i64] }:$src1, VR512:{ *:[v8i64] }:$src2)  =>  (VPMULLQZrr:{ *:[v8i64] } VR512:{ *:[v8i64] }:$src1, VR512:{ *:[v8i64] }:$src2)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VPMULLQZrr,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 4184,
      GIR_Done,
    // Label 217: @3446
    GIM_Reject,
    // Label 188: @3447
    GIM_Try, /*On fail goto*//*Label 218*/ 3504,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v16s16,
      GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v16s16,
      GIM_Try, /*On fail goto*//*Label 219*/ 3480, // Rule ID 1853 //
        GIM_CheckFeatures, GIFBS_HasAVX2_NoVLX_Or_NoBWI,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR256RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR256RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR256RegClassID,
        // (mul:{ *:[v16i16] } VR256:{ *:[v16i16] }:$src1, VR256:{ *:[v16i16] }:$src2)  =>  (VPMULLWYrr:{ *:[v16i16] } VR256:{ *:[v16i16] }:$src1, VR256:{ *:[v16i16] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VPMULLWYrr,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 1853,
        GIR_Done,
      // Label 219: @3480
      GIM_Try, /*On fail goto*//*Label 220*/ 3503, // Rule ID 4172 //
        GIM_CheckFeatures, GIFBS_HasBWI_HasVLX,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR256XRegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR256XRegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR256XRegClassID,
        // (mul:{ *:[v16i16] } VR256X:{ *:[v16i16] }:$src1, VR256X:{ *:[v16i16] }:$src2)  =>  (VPMULLWZ256rr:{ *:[v16i16] } VR256X:{ *:[v16i16] }:$src1, VR256X:{ *:[v16i16] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VPMULLWZ256rr,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 4172,
        GIR_Done,
      // Label 220: @3503
      GIM_Reject,
    // Label 218: @3504
    GIM_Reject,
    // Label 189: @3505
    GIM_Try, /*On fail goto*//*Label 221*/ 3536, // Rule ID 4139 //
      GIM_CheckFeatures, GIFBS_HasAVX512,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v16s32,
      GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v16s32,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR512RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR512RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR512RegClassID,
      // (mul:{ *:[v16i32] } VR512:{ *:[v16i32] }:$src1, VR512:{ *:[v16i32] }:$src2)  =>  (VPMULLDZrr:{ *:[v16i32] } VR512:{ *:[v16i32] }:$src1, VR512:{ *:[v16i32] }:$src2)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VPMULLDZrr,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 4139,
      GIR_Done,
    // Label 221: @3536
    GIM_Reject,
    // Label 190: @3537
    GIM_Try, /*On fail goto*//*Label 222*/ 3568, // Rule ID 4166 //
      GIM_CheckFeatures, GIFBS_HasBWI,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v32s16,
      GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v32s16,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR512RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR512RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR512RegClassID,
      // (mul:{ *:[v32i16] } VR512:{ *:[v32i16] }:$src1, VR512:{ *:[v32i16] }:$src2)  =>  (VPMULLWZrr:{ *:[v32i16] } VR512:{ *:[v32i16] }:$src1, VR512:{ *:[v32i16] }:$src2)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VPMULLWZrr,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 4166,
      GIR_Done,
    // Label 222: @3568
    GIM_Reject,
    // Label 191: @3569
    GIM_Reject,
    // Label 3: @3570
    GIM_SwitchType, /*MI*/0, /*Op*/0, /*[*/0, 24, /*)*//*default:*//*Label 237*/ 6681,
    /*GILLT_s1*//*Label 223*/ 3600,
    /*GILLT_s8*//*Label 224*/ 3898,
    /*GILLT_s16*//*Label 225*/ 3964,
    /*GILLT_s32*//*Label 226*/ 4063,
    /*GILLT_s64*//*Label 227*/ 5224, 0, 0,
    /*GILLT_v2s1*//*Label 228*/ 6096,
    /*GILLT_v2s64*//*Label 229*/ 6189,
    /*GILLT_v4s1*//*Label 230*/ 6270, 0,
    /*GILLT_v4s64*//*Label 231*/ 6363,
    /*GILLT_v8s1*//*Label 232*/ 6444, 0, 0,
    /*GILLT_v8s64*//*Label 233*/ 6553,
    /*GILLT_v16s1*//*Label 234*/ 6585, 0, 0, 0,
    /*GILLT_v32s1*//*Label 235*/ 6617, 0, 0,
    /*GILLT_v64s1*//*Label 236*/ 6649,
    // Label 223: @3600
    GIM_Try, /*On fail goto*//*Label 238*/ 3897,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_s1,
      GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_s1,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VK1RegClassID,
      GIM_Try, /*On fail goto*//*Label 239*/ 3715, // Rule ID 13735 //
        GIM_RecordInsn, /*DefineMI*/1, /*MI*/0, /*OpIdx*/1, // MIs[1]
        GIM_CheckOpcode, /*MI*/1, TargetOpcode::G_XOR,
        GIM_CheckType, /*MI*/1, /*Op*/1, /*Type*/GILLT_s1,
        GIM_CheckType, /*MI*/1, /*Op*/2, /*Type*/GILLT_s1,
        GIM_CheckRegBankForClass, /*MI*/1, /*Op*/1, /*RC*/X86::VK1RegClassID,
        GIM_CheckConstantInt, /*MI*/1, /*Op*/2, -1,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VK1RegClassID,
        GIM_CheckIsSafeToFold, /*InsnID*/1,
        // (and:{ *:[v1i1] } (xor:{ *:[v1i1] } VK1:{ *:[v1i1] }:$src1, -1:{ *:[v1i1] }), VK1:{ *:[v1i1] }:$src2)  =>  (COPY_TO_REGCLASS:{ *:[v1i1] } (KANDNWrr:{ *:[v16i1] } (COPY_TO_REGCLASS:{ *:[v16i1] } VK1:{ *:[v1i1] }:$src1, VK16:{ *:[i32] }), (COPY_TO_REGCLASS:{ *:[v16i1] } VK1:{ *:[v1i1] }:$src2, VK16:{ *:[i32] })), VK1:{ *:[i32] })
        GIR_MakeTempReg, /*TempRegID*/0, /*TypeID*/GILLT_v16s1,
        GIR_MakeTempReg, /*TempRegID*/1, /*TypeID*/GILLT_v16s1,
        GIR_MakeTempReg, /*TempRegID*/2, /*TypeID*/GILLT_v16s1,
        GIR_BuildMI, /*InsnID*/3, /*Opcode*/TargetOpcode::COPY,
        GIR_AddTempRegister, /*InsnID*/3, /*TempRegID*/2, /*TempRegFlags*/RegState::Define,
        GIR_Copy, /*NewInsnID*/3, /*OldInsnID*/0, /*OpIdx*/2, // src2
        GIR_ConstrainSelectedInstOperands, /*InsnID*/3,
        GIR_BuildMI, /*InsnID*/2, /*Opcode*/TargetOpcode::COPY,
        GIR_AddTempRegister, /*InsnID*/2, /*TempRegID*/1, /*TempRegFlags*/RegState::Define,
        GIR_Copy, /*NewInsnID*/2, /*OldInsnID*/1, /*OpIdx*/1, // src1
        GIR_ConstrainSelectedInstOperands, /*InsnID*/2,
        GIR_BuildMI, /*InsnID*/1, /*Opcode*/X86::KANDNWrr,
        GIR_AddTempRegister, /*InsnID*/1, /*TempRegID*/0, /*TempRegFlags*/RegState::Define,
        GIR_AddTempRegister, /*InsnID*/1, /*TempRegID*/1, /*TempRegFlags*/0,
        GIR_AddTempRegister, /*InsnID*/1, /*TempRegID*/2, /*TempRegFlags*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/1,
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/TargetOpcode::COPY,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_AddTempRegister, /*InsnID*/0, /*TempRegID*/0, /*TempRegFlags*/0,
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainOperandRC, /*InsnID*/0, /*Op*/0, /*RC VK1*/8,
        // GIR_Coverage, 13735,
        GIR_Done,
      // Label 239: @3715
      GIM_Try, /*On fail goto*//*Label 240*/ 3816, // Rule ID 18285 //
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VK1RegClassID,
        GIM_RecordInsn, /*DefineMI*/1, /*MI*/0, /*OpIdx*/2, // MIs[1]
        GIM_CheckOpcode, /*MI*/1, TargetOpcode::G_XOR,
        GIM_CheckType, /*MI*/1, /*Op*/1, /*Type*/GILLT_s1,
        GIM_CheckType, /*MI*/1, /*Op*/2, /*Type*/GILLT_s1,
        GIM_CheckRegBankForClass, /*MI*/1, /*Op*/1, /*RC*/X86::VK1RegClassID,
        GIM_CheckConstantInt, /*MI*/1, /*Op*/2, -1,
        GIM_CheckIsSafeToFold, /*InsnID*/1,
        // (and:{ *:[v1i1] } VK1:{ *:[v1i1] }:$src2, (xor:{ *:[v1i1] } VK1:{ *:[v1i1] }:$src1, -1:{ *:[v1i1] }))  =>  (COPY_TO_REGCLASS:{ *:[v1i1] } (KANDNWrr:{ *:[v16i1] } (COPY_TO_REGCLASS:{ *:[v16i1] } VK1:{ *:[v1i1] }:$src1, VK16:{ *:[i32] }), (COPY_TO_REGCLASS:{ *:[v16i1] } VK1:{ *:[v1i1] }:$src2, VK16:{ *:[i32] })), VK1:{ *:[i32] })
        GIR_MakeTempReg, /*TempRegID*/0, /*TypeID*/GILLT_v16s1,
        GIR_MakeTempReg, /*TempRegID*/1, /*TypeID*/GILLT_v16s1,
        GIR_MakeTempReg, /*TempRegID*/2, /*TypeID*/GILLT_v16s1,
        GIR_BuildMI, /*InsnID*/3, /*Opcode*/TargetOpcode::COPY,
        GIR_AddTempRegister, /*InsnID*/3, /*TempRegID*/2, /*TempRegFlags*/RegState::Define,
        GIR_Copy, /*NewInsnID*/3, /*OldInsnID*/0, /*OpIdx*/1, // src2
        GIR_ConstrainSelectedInstOperands, /*InsnID*/3,
        GIR_BuildMI, /*InsnID*/2, /*Opcode*/TargetOpcode::COPY,
        GIR_AddTempRegister, /*InsnID*/2, /*TempRegID*/1, /*TempRegFlags*/RegState::Define,
        GIR_Copy, /*NewInsnID*/2, /*OldInsnID*/1, /*OpIdx*/1, // src1
        GIR_ConstrainSelectedInstOperands, /*InsnID*/2,
        GIR_BuildMI, /*InsnID*/1, /*Opcode*/X86::KANDNWrr,
        GIR_AddTempRegister, /*InsnID*/1, /*TempRegID*/0, /*TempRegFlags*/RegState::Define,
        GIR_AddTempRegister, /*InsnID*/1, /*TempRegID*/1, /*TempRegFlags*/0,
        GIR_AddTempRegister, /*InsnID*/1, /*TempRegID*/2, /*TempRegFlags*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/1,
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/TargetOpcode::COPY,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_AddTempRegister, /*InsnID*/0, /*TempRegID*/0, /*TempRegFlags*/0,
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainOperandRC, /*InsnID*/0, /*Op*/0, /*RC VK1*/8,
        // GIR_Coverage, 18285,
        GIR_Done,
      // Label 240: @3816
      GIM_Try, /*On fail goto*//*Label 241*/ 3896, // Rule ID 13731 //
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VK1RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VK1RegClassID,
        // (and:{ *:[v1i1] } VK1:{ *:[v1i1] }:$src1, VK1:{ *:[v1i1] }:$src2)  =>  (COPY_TO_REGCLASS:{ *:[v1i1] } (KANDWrr:{ *:[v16i1] } (COPY_TO_REGCLASS:{ *:[v16i1] } VK1:{ *:[v1i1] }:$src1, VK16:{ *:[i32] }), (COPY_TO_REGCLASS:{ *:[v16i1] } VK1:{ *:[v1i1] }:$src2, VK16:{ *:[i32] })), VK1:{ *:[i32] })
        GIR_MakeTempReg, /*TempRegID*/0, /*TypeID*/GILLT_v16s1,
        GIR_MakeTempReg, /*TempRegID*/1, /*TypeID*/GILLT_v16s1,
        GIR_MakeTempReg, /*TempRegID*/2, /*TypeID*/GILLT_v16s1,
        GIR_BuildMI, /*InsnID*/3, /*Opcode*/TargetOpcode::COPY,
        GIR_AddTempRegister, /*InsnID*/3, /*TempRegID*/2, /*TempRegFlags*/RegState::Define,
        GIR_Copy, /*NewInsnID*/3, /*OldInsnID*/0, /*OpIdx*/2, // src2
        GIR_ConstrainSelectedInstOperands, /*InsnID*/3,
        GIR_BuildMI, /*InsnID*/2, /*Opcode*/TargetOpcode::COPY,
        GIR_AddTempRegister, /*InsnID*/2, /*TempRegID*/1, /*TempRegFlags*/RegState::Define,
        GIR_Copy, /*NewInsnID*/2, /*OldInsnID*/0, /*OpIdx*/1, // src1
        GIR_ConstrainSelectedInstOperands, /*InsnID*/2,
        GIR_BuildMI, /*InsnID*/1, /*Opcode*/X86::KANDWrr,
        GIR_AddTempRegister, /*InsnID*/1, /*TempRegID*/0, /*TempRegFlags*/RegState::Define,
        GIR_AddTempRegister, /*InsnID*/1, /*TempRegID*/1, /*TempRegFlags*/0,
        GIR_AddTempRegister, /*InsnID*/1, /*TempRegID*/2, /*TempRegFlags*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/1,
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/TargetOpcode::COPY,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_AddTempRegister, /*InsnID*/0, /*TempRegID*/0, /*TempRegFlags*/0,
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainOperandRC, /*InsnID*/0, /*Op*/0, /*RC VK1*/8,
        // GIR_Coverage, 13731,
        GIR_Done,
      // Label 241: @3896
      GIM_Reject,
    // Label 238: @3897
    GIM_Reject,
    // Label 224: @3898
    GIM_Try, /*On fail goto*//*Label 242*/ 3963,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_s8,
      GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_s8,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::GR8RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::GR8RegClassID,
      GIM_Try, /*On fail goto*//*Label 243*/ 3946, // Rule ID 16194 //
        GIM_RecordInsn, /*DefineMI*/1, /*MI*/0, /*OpIdx*/2, // MIs[1]
        GIM_CheckOpcode, /*MI*/1, TargetOpcode::G_CONSTANT,
        // MIs[1] Operand 1
        // No operand predicates
        GIM_CheckIsSafeToFold, /*InsnID*/1,
        // (and:{ *:[i8] } GR8:{ *:[i8] }:$src1, (imm:{ *:[i8] }):$src2)  =>  (AND8ri:{ *:[i8] }:{ *:[i32] } GR8:{ *:[i8] }:$src1, (imm:{ *:[i8] }):$src2)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::AND8ri,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // src1
        GIR_CopyConstantAsSImm, /*NewInsnID*/0, /*OldInsnID*/1, // src2
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 16194,
        GIR_Done,
      // Label 243: @3946
      GIM_Try, /*On fail goto*//*Label 244*/ 3962, // Rule ID 16186 //
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::GR8RegClassID,
        // (and:{ *:[i8] } GR8:{ *:[i8] }:$src1, GR8:{ *:[i8] }:$src2)  =>  (AND8rr:{ *:[i8] }:{ *:[i32] } GR8:{ *:[i8] }:$src1, GR8:{ *:[i8] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::AND8rr,
        GIR_AddImplicitDef, /*InsnID*/0, X86::EFLAGS,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 16186,
        GIR_Done,
      // Label 244: @3962
      GIM_Reject,
    // Label 242: @3963
    GIM_Reject,
    // Label 225: @3964
    GIM_Try, /*On fail goto*//*Label 245*/ 4062,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_s16,
      GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_s16,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::GR16RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::GR16RegClassID,
      GIM_Try, /*On fail goto*//*Label 246*/ 4015, // Rule ID 16197 //
        GIM_RecordInsn, /*DefineMI*/1, /*MI*/0, /*OpIdx*/2, // MIs[1]
        GIM_CheckOpcode, /*MI*/1, TargetOpcode::G_CONSTANT,
        GIM_CheckI64ImmPredicate, /*MI*/1, /*Predicate*/GIPFP_I64_Predicate_i16immSExt8,
        // MIs[1] Operand 1
        // No operand predicates
        GIM_CheckIsSafeToFold, /*InsnID*/1,
        // (and:{ *:[i16] } GR16:{ *:[i16] }:$src1, (imm:{ *:[i16] })<<P:Predicate_i16immSExt8>>:$src2)  =>  (AND16ri8:{ *:[i16] }:{ *:[i32] } GR16:{ *:[i16] }:$src1, (imm:{ *:[i16] })<<P:Predicate_i16immSExt8>>:$src2)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::AND16ri8,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // src1
        GIR_CopyConstantAsSImm, /*NewInsnID*/0, /*OldInsnID*/1, // src2
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 16197,
        GIR_Done,
      // Label 246: @4015
      GIM_Try, /*On fail goto*//*Label 247*/ 4045, // Rule ID 16195 //
        GIM_RecordInsn, /*DefineMI*/1, /*MI*/0, /*OpIdx*/2, // MIs[1]
        GIM_CheckOpcode, /*MI*/1, TargetOpcode::G_CONSTANT,
        // MIs[1] Operand 1
        // No operand predicates
        GIM_CheckIsSafeToFold, /*InsnID*/1,
        // (and:{ *:[i16] } GR16:{ *:[i16] }:$src1, (imm:{ *:[i16] }):$src2)  =>  (AND16ri:{ *:[i16] }:{ *:[i32] } GR16:{ *:[i16] }:$src1, (imm:{ *:[i16] }):$src2)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::AND16ri,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // src1
        GIR_CopyConstantAsSImm, /*NewInsnID*/0, /*OldInsnID*/1, // src2
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 16195,
        GIR_Done,
      // Label 247: @4045
      GIM_Try, /*On fail goto*//*Label 248*/ 4061, // Rule ID 16187 //
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::GR16RegClassID,
        // (and:{ *:[i16] } GR16:{ *:[i16] }:$src1, GR16:{ *:[i16] }:$src2)  =>  (AND16rr:{ *:[i16] }:{ *:[i32] } GR16:{ *:[i16] }:$src1, GR16:{ *:[i16] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::AND16rr,
        GIR_AddImplicitDef, /*InsnID*/0, X86::EFLAGS,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 16187,
        GIR_Done,
      // Label 248: @4061
      GIM_Reject,
    // Label 245: @4062
    GIM_Reject,
    // Label 226: @4063
    GIM_Try, /*On fail goto*//*Label 249*/ 5223,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_s32,
      GIM_Try, /*On fail goto*//*Label 250*/ 4164, // Rule ID 18144 //
        GIM_CheckFeatures, GIFBS_HasBMI2,
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_s32,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::GR32RegClassID,
        GIM_RecordInsn, /*DefineMI*/1, /*MI*/0, /*OpIdx*/1, // MIs[1]
        GIM_CheckOpcode, /*MI*/1, TargetOpcode::G_LSHR,
        GIM_CheckType, /*MI*/1, /*Op*/1, /*Type*/GILLT_s32,
        GIM_CheckType, /*MI*/1, /*Op*/2, /*Type*/GILLT_s8,
        GIM_CheckConstantInt, /*MI*/1, /*Op*/1, -1,
        GIM_RecordInsn, /*DefineMI*/2, /*MI*/1, /*OpIdx*/2, // MIs[2]
        GIM_CheckOpcode, /*MI*/2, TargetOpcode::G_TRUNC,
        GIM_CheckType, /*MI*/2, /*Op*/1, /*Type*/GILLT_s32,
        GIM_RecordInsn, /*DefineMI*/3, /*MI*/2, /*OpIdx*/1, // MIs[3]
        GIM_CheckOpcode, /*MI*/3, TargetOpcode::G_SUB,
        GIM_CheckType, /*MI*/3, /*Op*/1, /*Type*/GILLT_s32,
        GIM_CheckType, /*MI*/3, /*Op*/2, /*Type*/GILLT_s32,
        GIM_CheckConstantInt, /*MI*/3, /*Op*/1, 32,
        GIM_CheckRegBankForClass, /*MI*/3, /*Op*/2, /*RC*/X86::GR32RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::GR32RegClassID,
        GIM_CheckIsSafeToFold, /*InsnID*/1,
        GIM_CheckIsSafeToFold, /*InsnID*/2,
        GIM_CheckIsSafeToFold, /*InsnID*/3,
        // (and:{ *:[i32] } (srl:{ *:[i32] } -1:{ *:[i32] }, (trunc:{ *:[i8] } (sub:{ *:[i32] } 32:{ *:[i32] }, GR32:{ *:[i32] }:$lz))), GR32:{ *:[i32] }:$src)  =>  (BZHI32rr:{ *:[i32] }:{ *:[i32] } GR32:{ *:[i32] }:$src, GR32:{ *:[i32] }:$lz)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::BZHI32rr,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/2, // src
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/3, /*OpIdx*/2, // lz
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 18144,
        GIR_Done,
      // Label 250: @4164
      GIM_Try, /*On fail goto*//*Label 251*/ 4259, // Rule ID 12211 //
        GIM_CheckFeatures, GIFBS_HasBMI2,
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_s32,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::GR32RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::GR32RegClassID,
        GIM_RecordInsn, /*DefineMI*/1, /*MI*/0, /*OpIdx*/2, // MIs[1]
        GIM_CheckOpcode, /*MI*/1, TargetOpcode::G_LSHR,
        GIM_CheckType, /*MI*/1, /*Op*/1, /*Type*/GILLT_s32,
        GIM_CheckType, /*MI*/1, /*Op*/2, /*Type*/GILLT_s8,
        GIM_CheckConstantInt, /*MI*/1, /*Op*/1, -1,
        GIM_RecordInsn, /*DefineMI*/2, /*MI*/1, /*OpIdx*/2, // MIs[2]
        GIM_CheckOpcode, /*MI*/2, TargetOpcode::G_TRUNC,
        GIM_CheckType, /*MI*/2, /*Op*/1, /*Type*/GILLT_s32,
        GIM_RecordInsn, /*DefineMI*/3, /*MI*/2, /*OpIdx*/1, // MIs[3]
        GIM_CheckOpcode, /*MI*/3, TargetOpcode::G_SUB,
        GIM_CheckType, /*MI*/3, /*Op*/1, /*Type*/GILLT_s32,
        GIM_CheckType, /*MI*/3, /*Op*/2, /*Type*/GILLT_s32,
        GIM_CheckConstantInt, /*MI*/3, /*Op*/1, 32,
        GIM_CheckRegBankForClass, /*MI*/3, /*Op*/2, /*RC*/X86::GR32RegClassID,
        GIM_CheckIsSafeToFold, /*InsnID*/1,
        GIM_CheckIsSafeToFold, /*InsnID*/2,
        GIM_CheckIsSafeToFold, /*InsnID*/3,
        // (and:{ *:[i32] } GR32:{ *:[i32] }:$src, (srl:{ *:[i32] } -1:{ *:[i32] }, (trunc:{ *:[i8] } (sub:{ *:[i32] } 32:{ *:[i32] }, GR32:{ *:[i32] }:$lz))))  =>  (BZHI32rr:{ *:[i32] }:{ *:[i32] } GR32:{ *:[i32] }:$src, GR32:{ *:[i32] }:$lz)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::BZHI32rr,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // src
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/3, /*OpIdx*/2, // lz
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 12211,
        GIR_Done,
      // Label 251: @4259
      GIM_Try, /*On fail goto*//*Label 252*/ 4334, // Rule ID 18154 //
        GIM_CheckFeatures, GIFBS_HasTBM,
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_s32,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::GR32RegClassID,
        GIM_RecordInsn, /*DefineMI*/1, /*MI*/0, /*OpIdx*/1, // MIs[1]
        GIM_CheckOpcode, /*MI*/1, TargetOpcode::G_ADD,
        GIM_CheckType, /*MI*/1, /*Op*/1, /*Type*/GILLT_s32,
        GIM_CheckType, /*MI*/1, /*Op*/2, /*Type*/GILLT_s32,
        GIM_CheckRegBankForClass, /*MI*/1, /*Op*/1, /*RC*/X86::GR32RegClassID,
        GIM_CheckConstantInt, /*MI*/1, /*Op*/2, 1,
        GIM_RecordInsn, /*DefineMI*/2, /*MI*/0, /*OpIdx*/2, // MIs[2]
        GIM_CheckOpcode, /*MI*/2, TargetOpcode::G_XOR,
        GIM_CheckType, /*MI*/2, /*Op*/2, /*Type*/GILLT_s32,
        // MIs[2] src
        GIM_CheckIsSameOperand, /*MI*/2, /*OpIdx*/1, /*OtherMI*/1, /*OtherOpIdx*/1,
        GIM_CheckConstantInt, /*MI*/2, /*Op*/2, -1,
        GIM_CheckIsSafeToFold, /*InsnID*/1,
        GIM_CheckIsSafeToFold, /*InsnID*/2,
        // (and:{ *:[i32] } (add:{ *:[i32] } GR32:{ *:[i32] }:$src, 1:{ *:[i32] }), (xor:{ *:[i32] } GR32:{ *:[i32] }:$src, -1:{ *:[i32] }))  =>  (BLCIC32rr:{ *:[i32] }:{ *:[i32] } GR32:{ *:[i32] }:$src)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::BLCIC32rr,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/1, /*OpIdx*/1, // src
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 18154,
        GIR_Done,
      // Label 252: @4334
      GIM_Try, /*On fail goto*//*Label 253*/ 4409, // Rule ID 18166 //
        GIM_CheckFeatures, GIFBS_HasTBM,
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_s32,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::GR32RegClassID,
        GIM_RecordInsn, /*DefineMI*/1, /*MI*/0, /*OpIdx*/1, // MIs[1]
        GIM_CheckOpcode, /*MI*/1, TargetOpcode::G_ADD,
        GIM_CheckType, /*MI*/1, /*Op*/1, /*Type*/GILLT_s32,
        GIM_CheckType, /*MI*/1, /*Op*/2, /*Type*/GILLT_s32,
        GIM_CheckRegBankForClass, /*MI*/1, /*Op*/1, /*RC*/X86::GR32RegClassID,
        GIM_CheckConstantInt, /*MI*/1, /*Op*/2, -1,
        GIM_RecordInsn, /*DefineMI*/2, /*MI*/0, /*OpIdx*/2, // MIs[2]
        GIM_CheckOpcode, /*MI*/2, TargetOpcode::G_XOR,
        GIM_CheckType, /*MI*/2, /*Op*/2, /*Type*/GILLT_s32,
        // MIs[2] src
        GIM_CheckIsSameOperand, /*MI*/2, /*OpIdx*/1, /*OtherMI*/1, /*OtherOpIdx*/1,
        GIM_CheckConstantInt, /*MI*/2, /*Op*/2, -1,
        GIM_CheckIsSafeToFold, /*InsnID*/1,
        GIM_CheckIsSafeToFold, /*InsnID*/2,
        // (and:{ *:[i32] } (add:{ *:[i32] } GR32:{ *:[i32] }:$src, -1:{ *:[i32] }), (xor:{ *:[i32] } GR32:{ *:[i32] }:$src, -1:{ *:[i32] }))  =>  (TZMSK32rr:{ *:[i32] }:{ *:[i32] } GR32:{ *:[i32] }:$src)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::TZMSK32rr,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/1, /*OpIdx*/1, // src
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 18166,
        GIR_Done,
      // Label 253: @4409
      GIM_Try, /*On fail goto*//*Label 254*/ 4484, // Rule ID 12227 //
        GIM_CheckFeatures, GIFBS_HasTBM,
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_s32,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::GR32RegClassID,
        GIM_RecordInsn, /*DefineMI*/1, /*MI*/0, /*OpIdx*/1, // MIs[1]
        GIM_CheckOpcode, /*MI*/1, TargetOpcode::G_XOR,
        GIM_CheckType, /*MI*/1, /*Op*/1, /*Type*/GILLT_s32,
        GIM_CheckType, /*MI*/1, /*Op*/2, /*Type*/GILLT_s32,
        GIM_CheckRegBankForClass, /*MI*/1, /*Op*/1, /*RC*/X86::GR32RegClassID,
        GIM_CheckConstantInt, /*MI*/1, /*Op*/2, -1,
        GIM_RecordInsn, /*DefineMI*/2, /*MI*/0, /*OpIdx*/2, // MIs[2]
        GIM_CheckOpcode, /*MI*/2, TargetOpcode::G_ADD,
        GIM_CheckType, /*MI*/2, /*Op*/2, /*Type*/GILLT_s32,
        // MIs[2] src
        GIM_CheckIsSameOperand, /*MI*/2, /*OpIdx*/1, /*OtherMI*/1, /*OtherOpIdx*/1,
        GIM_CheckConstantInt, /*MI*/2, /*Op*/2, 1,
        GIM_CheckIsSafeToFold, /*InsnID*/1,
        GIM_CheckIsSafeToFold, /*InsnID*/2,
        // (and:{ *:[i32] } (xor:{ *:[i32] } GR32:{ *:[i32] }:$src, -1:{ *:[i32] }), (add:{ *:[i32] } GR32:{ *:[i32] }:$src, 1:{ *:[i32] }))  =>  (BLCIC32rr:{ *:[i32] }:{ *:[i32] } GR32:{ *:[i32] }:$src)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::BLCIC32rr,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/1, /*OpIdx*/1, // src
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 12227,
        GIR_Done,
      // Label 254: @4484
      GIM_Try, /*On fail goto*//*Label 255*/ 4559, // Rule ID 12239 //
        GIM_CheckFeatures, GIFBS_HasTBM,
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_s32,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::GR32RegClassID,
        GIM_RecordInsn, /*DefineMI*/1, /*MI*/0, /*OpIdx*/1, // MIs[1]
        GIM_CheckOpcode, /*MI*/1, TargetOpcode::G_XOR,
        GIM_CheckType, /*MI*/1, /*Op*/1, /*Type*/GILLT_s32,
        GIM_CheckType, /*MI*/1, /*Op*/2, /*Type*/GILLT_s32,
        GIM_CheckRegBankForClass, /*MI*/1, /*Op*/1, /*RC*/X86::GR32RegClassID,
        GIM_CheckConstantInt, /*MI*/1, /*Op*/2, -1,
        GIM_RecordInsn, /*DefineMI*/2, /*MI*/0, /*OpIdx*/2, // MIs[2]
        GIM_CheckOpcode, /*MI*/2, TargetOpcode::G_ADD,
        GIM_CheckType, /*MI*/2, /*Op*/2, /*Type*/GILLT_s32,
        // MIs[2] src
        GIM_CheckIsSameOperand, /*MI*/2, /*OpIdx*/1, /*OtherMI*/1, /*OtherOpIdx*/1,
        GIM_CheckConstantInt, /*MI*/2, /*Op*/2, -1,
        GIM_CheckIsSafeToFold, /*InsnID*/1,
        GIM_CheckIsSafeToFold, /*InsnID*/2,
        // (and:{ *:[i32] } (xor:{ *:[i32] } GR32:{ *:[i32] }:$src, -1:{ *:[i32] }), (add:{ *:[i32] } GR32:{ *:[i32] }:$src, -1:{ *:[i32] }))  =>  (TZMSK32rr:{ *:[i32] }:{ *:[i32] } GR32:{ *:[i32] }:$src)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::TZMSK32rr,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/1, /*OpIdx*/1, // src
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 12239,
        GIR_Done,
      // Label 255: @4559
      GIM_Try, /*On fail goto*//*Label 256*/ 4613, // Rule ID 18148 //
        GIM_CheckFeatures, GIFBS_HasTBM,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::GR32RegClassID,
        GIM_RecordInsn, /*DefineMI*/1, /*MI*/0, /*OpIdx*/1, // MIs[1]
        GIM_CheckOpcode, /*MI*/1, TargetOpcode::G_ADD,
        GIM_CheckType, /*MI*/1, /*Op*/1, /*Type*/GILLT_s32,
        GIM_CheckType, /*MI*/1, /*Op*/2, /*Type*/GILLT_s32,
        GIM_CheckRegBankForClass, /*MI*/1, /*Op*/1, /*RC*/X86::GR32RegClassID,
        GIM_CheckConstantInt, /*MI*/1, /*Op*/2, 1,
        // MIs[0] src
        GIM_CheckIsSameOperand, /*MI*/0, /*OpIdx*/2, /*OtherMI*/1, /*OtherOpIdx*/1,
        GIM_CheckIsSafeToFold, /*InsnID*/1,
        // (and:{ *:[i32] } (add:{ *:[i32] } GR32:{ *:[i32] }:$src, 1:{ *:[i32] }), GR32:{ *:[i32] }:$src)  =>  (BLCFILL32rr:{ *:[i32] }:{ *:[i32] } GR32:{ *:[i32] }:$src)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::BLCFILL32rr,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/1, /*OpIdx*/1, // src
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 18148,
        GIR_Done,
      // Label 256: @4613
      GIM_Try, /*On fail goto*//*Label 257*/ 4667, // Rule ID 18674 //
        GIM_CheckFeatures, GIFBS_HasBMI,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::GR32RegClassID,
        GIM_RecordInsn, /*DefineMI*/1, /*MI*/0, /*OpIdx*/1, // MIs[1]
        GIM_CheckOpcode, /*MI*/1, TargetOpcode::G_ADD,
        GIM_CheckType, /*MI*/1, /*Op*/1, /*Type*/GILLT_s32,
        GIM_CheckType, /*MI*/1, /*Op*/2, /*Type*/GILLT_s32,
        GIM_CheckRegBankForClass, /*MI*/1, /*Op*/1, /*RC*/X86::GR32RegClassID,
        GIM_CheckConstantInt, /*MI*/1, /*Op*/2, -1,
        // MIs[0] src
        GIM_CheckIsSameOperand, /*MI*/0, /*OpIdx*/2, /*OtherMI*/1, /*OtherOpIdx*/1,
        GIM_CheckIsSafeToFold, /*InsnID*/1,
        // (and:{ *:[i32] } (add:{ *:[i32] } GR32:{ *:[i32] }:$src, -1:{ *:[i32] }), GR32:{ *:[i32] }:$src)  =>  (BLSR32rr:{ *:[i32] }:{ *:[i32] } GR32:{ *:[i32] }:$src)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::BLSR32rr,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/1, /*OpIdx*/1, // src
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 18674,
        GIR_Done,
      // Label 257: @4667
      GIM_Try, /*On fail goto*//*Label 258*/ 4721, // Rule ID 18678 //
        GIM_CheckFeatures, GIFBS_HasBMI,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::GR32RegClassID,
        GIM_RecordInsn, /*DefineMI*/1, /*MI*/0, /*OpIdx*/1, // MIs[1]
        GIM_CheckOpcode, /*MI*/1, TargetOpcode::G_SUB,
        GIM_CheckType, /*MI*/1, /*Op*/1, /*Type*/GILLT_s32,
        GIM_CheckType, /*MI*/1, /*Op*/2, /*Type*/GILLT_s32,
        GIM_CheckConstantInt, /*MI*/1, /*Op*/1, 0,
        GIM_CheckRegBankForClass, /*MI*/1, /*Op*/2, /*RC*/X86::GR32RegClassID,
        // MIs[0] src
        GIM_CheckIsSameOperand, /*MI*/0, /*OpIdx*/2, /*OtherMI*/1, /*OtherOpIdx*/2,
        GIM_CheckIsSafeToFold, /*InsnID*/1,
        // (and:{ *:[i32] } (sub:{ *:[i32] } 0:{ *:[i32] }, GR32:{ *:[i32] }:$src), GR32:{ *:[i32] }:$src)  =>  (BLSI32rr:{ *:[i32] }:{ *:[i32] } GR32:{ *:[i32] }:$src)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::BLSI32rr,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/1, /*OpIdx*/2, // src
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 18678,
        GIR_Done,
      // Label 258: @4721
      GIM_Try, /*On fail goto*//*Label 259*/ 4775, // Rule ID 12221 //
        GIM_CheckFeatures, GIFBS_HasTBM,
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_s32,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::GR32RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::GR32RegClassID,
        GIM_RecordInsn, /*DefineMI*/1, /*MI*/0, /*OpIdx*/2, // MIs[1]
        GIM_CheckOpcode, /*MI*/1, TargetOpcode::G_ADD,
        GIM_CheckType, /*MI*/1, /*Op*/2, /*Type*/GILLT_s32,
        // MIs[1] src
        GIM_CheckIsSameOperand, /*MI*/1, /*OpIdx*/1, /*OtherMI*/0, /*OtherOpIdx*/1,
        GIM_CheckConstantInt, /*MI*/1, /*Op*/2, 1,
        GIM_CheckIsSafeToFold, /*InsnID*/1,
        // (and:{ *:[i32] } GR32:{ *:[i32] }:$src, (add:{ *:[i32] } GR32:{ *:[i32] }:$src, 1:{ *:[i32] }))  =>  (BLCFILL32rr:{ *:[i32] }:{ *:[i32] } GR32:{ *:[i32] }:$src)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::BLCFILL32rr,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // src
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 12221,
        GIR_Done,
      // Label 259: @4775
      GIM_Try, /*On fail goto*//*Label 260*/ 4829, // Rule ID 16393 //
        GIM_CheckFeatures, GIFBS_HasBMI,
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_s32,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::GR32RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::GR32RegClassID,
        GIM_RecordInsn, /*DefineMI*/1, /*MI*/0, /*OpIdx*/2, // MIs[1]
        GIM_CheckOpcode, /*MI*/1, TargetOpcode::G_ADD,
        GIM_CheckType, /*MI*/1, /*Op*/2, /*Type*/GILLT_s32,
        // MIs[1] src
        GIM_CheckIsSameOperand, /*MI*/1, /*OpIdx*/1, /*OtherMI*/0, /*OtherOpIdx*/1,
        GIM_CheckConstantInt, /*MI*/1, /*Op*/2, -1,
        GIM_CheckIsSafeToFold, /*InsnID*/1,
        // (and:{ *:[i32] } GR32:{ *:[i32] }:$src, (add:{ *:[i32] } GR32:{ *:[i32] }:$src, -1:{ *:[i32] }))  =>  (BLSR32rr:{ *:[i32] }:{ *:[i32] } GR32:{ *:[i32] }:$src)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::BLSR32rr,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // src
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 16393,
        GIR_Done,
      // Label 260: @4829
      GIM_Try, /*On fail goto*//*Label 261*/ 4883, // Rule ID 16397 //
        GIM_CheckFeatures, GIFBS_HasBMI,
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_s32,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::GR32RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::GR32RegClassID,
        GIM_RecordInsn, /*DefineMI*/1, /*MI*/0, /*OpIdx*/2, // MIs[1]
        GIM_CheckOpcode, /*MI*/1, TargetOpcode::G_SUB,
        GIM_CheckType, /*MI*/1, /*Op*/1, /*Type*/GILLT_s32,
        GIM_CheckConstantInt, /*MI*/1, /*Op*/1, 0,
        // MIs[1] src
        GIM_CheckIsSameOperand, /*MI*/1, /*OpIdx*/2, /*OtherMI*/0, /*OtherOpIdx*/1,
        GIM_CheckIsSafeToFold, /*InsnID*/1,
        // (and:{ *:[i32] } GR32:{ *:[i32] }:$src, (sub:{ *:[i32] } 0:{ *:[i32] }, GR32:{ *:[i32] }:$src))  =>  (BLSI32rr:{ *:[i32] }:{ *:[i32] } GR32:{ *:[i32] }:$src)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::BLSI32rr,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // src
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 16397,
        GIR_Done,
      // Label 261: @4883
      GIM_Try, /*On fail goto*//*Label 262*/ 4934, // Rule ID 15970 //
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_s32,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::GR32RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::GR32RegClassID,
        GIM_CheckConstantInt, /*MI*/0, /*Op*/2, 65535,
        // (and:{ *:[i32] } GR32:{ *:[i32] }:$src1, 65535:{ *:[i32] })  =>  (MOVZX32rr16:{ *:[i32] } (EXTRACT_SUBREG:{ *:[i16] } GR32:{ *:[i32] }:$src1, sub_16bit:{ *:[i32] }))
        GIR_MakeTempReg, /*TempRegID*/0, /*TypeID*/GILLT_s16,
        GIR_BuildMI, /*InsnID*/1, /*Opcode*/TargetOpcode::COPY,
        GIR_AddTempRegister, /*InsnID*/1, /*TempRegID*/0, /*TempRegFlags*/RegState::Define,
        GIR_CopySubReg, /*NewInsnID*/1, /*OldInsnID*/0, /*OpIdx*/1, /*SubRegIdx*/4, // src1
        GIR_ConstrainSelectedInstOperands, /*InsnID*/1,
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::MOVZX32rr16,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_AddTempRegister, /*InsnID*/0, /*TempRegID*/0, /*TempRegFlags*/0,
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 15970,
        GIR_Done,
      // Label 262: @4934
      GIM_Try, /*On fail goto*//*Label 263*/ 4985, // Rule ID 15971 //
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_s32,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::GR32RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::GR32RegClassID,
        GIM_CheckConstantInt, /*MI*/0, /*Op*/2, 255,
        // (and:{ *:[i32] } GR32:{ *:[i32] }:$src1, 255:{ *:[i32] })  =>  (MOVZX32rr8:{ *:[i32] } (EXTRACT_SUBREG:{ *:[i8] } GR32:{ *:[i32] }:$src1, sub_8bit:{ *:[i32] }))
        GIR_MakeTempReg, /*TempRegID*/0, /*TypeID*/GILLT_s8,
        GIR_BuildMI, /*InsnID*/1, /*Opcode*/TargetOpcode::COPY,
        GIR_AddTempRegister, /*InsnID*/1, /*TempRegID*/0, /*TempRegFlags*/RegState::Define,
        GIR_CopySubReg, /*NewInsnID*/1, /*OldInsnID*/0, /*OpIdx*/1, /*SubRegIdx*/1, // src1
        GIR_ConstrainSelectedInstOperands, /*InsnID*/1,
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::MOVZX32rr8,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_AddTempRegister, /*InsnID*/0, /*TempRegID*/0, /*TempRegFlags*/0,
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 15971,
        GIR_Done,
      // Label 263: @4985
      GIM_Try, /*On fail goto*//*Label 264*/ 5030, // Rule ID 16198 //
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_s32,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::GR32RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::GR32RegClassID,
        GIM_RecordInsn, /*DefineMI*/1, /*MI*/0, /*OpIdx*/2, // MIs[1]
        GIM_CheckOpcode, /*MI*/1, TargetOpcode::G_CONSTANT,
        GIM_CheckI64ImmPredicate, /*MI*/1, /*Predicate*/GIPFP_I64_Predicate_i32immSExt8,
        // MIs[1] Operand 1
        // No operand predicates
        GIM_CheckIsSafeToFold, /*InsnID*/1,
        // (and:{ *:[i32] } GR32:{ *:[i32] }:$src1, (imm:{ *:[i32] })<<P:Predicate_i32immSExt8>>:$src2)  =>  (AND32ri8:{ *:[i32] }:{ *:[i32] } GR32:{ *:[i32] }:$src1, (imm:{ *:[i32] })<<P:Predicate_i32immSExt8>>:$src2)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::AND32ri8,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // src1
        GIR_CopyConstantAsSImm, /*NewInsnID*/0, /*OldInsnID*/1, // src2
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 16198,
        GIR_Done,
      // Label 264: @5030
      GIM_Try, /*On fail goto*//*Label 265*/ 5072, // Rule ID 16196 //
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_s32,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::GR32RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::GR32RegClassID,
        GIM_RecordInsn, /*DefineMI*/1, /*MI*/0, /*OpIdx*/2, // MIs[1]
        GIM_CheckOpcode, /*MI*/1, TargetOpcode::G_CONSTANT,
        // MIs[1] Operand 1
        // No operand predicates
        GIM_CheckIsSafeToFold, /*InsnID*/1,
        // (and:{ *:[i32] } GR32:{ *:[i32] }:$src1, (imm:{ *:[i32] }):$src2)  =>  (AND32ri:{ *:[i32] }:{ *:[i32] } GR32:{ *:[i32] }:$src1, (imm:{ *:[i32] }):$src2)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::AND32ri,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // src1
        GIR_CopyConstantAsSImm, /*NewInsnID*/0, /*OldInsnID*/1, // src2
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 16196,
        GIR_Done,
      // Label 265: @5072
      GIM_Try, /*On fail goto*//*Label 266*/ 5133, // Rule ID 12249 //
        GIM_CheckFeatures, GIFBS_HasBMI,
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_s32,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::GR32RegClassID,
        GIM_RecordInsn, /*DefineMI*/1, /*MI*/0, /*OpIdx*/1, // MIs[1]
        GIM_CheckOpcode, /*MI*/1, TargetOpcode::G_XOR,
        GIM_CheckType, /*MI*/1, /*Op*/1, /*Type*/GILLT_s32,
        GIM_CheckType, /*MI*/1, /*Op*/2, /*Type*/GILLT_s32,
        GIM_CheckRegBankForClass, /*MI*/1, /*Op*/1, /*RC*/X86::GR32RegClassID,
        GIM_CheckConstantInt, /*MI*/1, /*Op*/2, -1,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::GR32RegClassID,
        GIM_CheckIsSafeToFold, /*InsnID*/1,
        // (and:{ *:[i32] } (xor:{ *:[i32] } GR32:{ *:[i32] }:$src1, -1:{ *:[i32] }), GR32:{ *:[i32] }:$src2)  =>  (ANDN32rr:{ *:[i32] }:{ *:[i32] } GR32:{ *:[i32] }:$src1, GR32:{ *:[i32] }:$src2)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::ANDN32rr,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/1, /*OpIdx*/1, // src1
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/2, // src2
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 12249,
        GIR_Done,
      // Label 266: @5133
      GIM_Try, /*On fail goto*//*Label 267*/ 5194, // Rule ID 18168 //
        GIM_CheckFeatures, GIFBS_HasBMI,
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_s32,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::GR32RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::GR32RegClassID,
        GIM_RecordInsn, /*DefineMI*/1, /*MI*/0, /*OpIdx*/2, // MIs[1]
        GIM_CheckOpcode, /*MI*/1, TargetOpcode::G_XOR,
        GIM_CheckType, /*MI*/1, /*Op*/1, /*Type*/GILLT_s32,
        GIM_CheckType, /*MI*/1, /*Op*/2, /*Type*/GILLT_s32,
        GIM_CheckRegBankForClass, /*MI*/1, /*Op*/1, /*RC*/X86::GR32RegClassID,
        GIM_CheckConstantInt, /*MI*/1, /*Op*/2, -1,
        GIM_CheckIsSafeToFold, /*InsnID*/1,
        // (and:{ *:[i32] } GR32:{ *:[i32] }:$src2, (xor:{ *:[i32] } GR32:{ *:[i32] }:$src1, -1:{ *:[i32] }))  =>  (ANDN32rr:{ *:[i32] }:{ *:[i32] } GR32:{ *:[i32] }:$src1, GR32:{ *:[i32] }:$src2)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::ANDN32rr,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/1, /*OpIdx*/1, // src1
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // src2
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 18168,
        GIR_Done,
      // Label 267: @5194
      GIM_Try, /*On fail goto*//*Label 268*/ 5222, // Rule ID 16188 //
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_s32,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::GR32RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::GR32RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::GR32RegClassID,
        // (and:{ *:[i32] } GR32:{ *:[i32] }:$src1, GR32:{ *:[i32] }:$src2)  =>  (AND32rr:{ *:[i32] }:{ *:[i32] } GR32:{ *:[i32] }:$src1, GR32:{ *:[i32] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::AND32rr,
        GIR_AddImplicitDef, /*InsnID*/0, X86::EFLAGS,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 16188,
        GIR_Done,
      // Label 268: @5222
      GIM_Reject,
    // Label 249: @5223
    GIM_Reject,
    // Label 227: @5224
    GIM_Try, /*On fail goto*//*Label 269*/ 6095,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_s64,
      GIM_Try, /*On fail goto*//*Label 270*/ 5305, // Rule ID 18155 //
        GIM_CheckFeatures, GIFBS_HasTBM,
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_s64,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::GR64RegClassID,
        GIM_RecordInsn, /*DefineMI*/1, /*MI*/0, /*OpIdx*/1, // MIs[1]
        GIM_CheckOpcode, /*MI*/1, TargetOpcode::G_ADD,
        GIM_CheckType, /*MI*/1, /*Op*/1, /*Type*/GILLT_s64,
        GIM_CheckType, /*MI*/1, /*Op*/2, /*Type*/GILLT_s64,
        GIM_CheckRegBankForClass, /*MI*/1, /*Op*/1, /*RC*/X86::GR64RegClassID,
        GIM_CheckConstantInt, /*MI*/1, /*Op*/2, 1,
        GIM_RecordInsn, /*DefineMI*/2, /*MI*/0, /*OpIdx*/2, // MIs[2]
        GIM_CheckOpcode, /*MI*/2, TargetOpcode::G_XOR,
        GIM_CheckType, /*MI*/2, /*Op*/2, /*Type*/GILLT_s64,
        // MIs[2] src
        GIM_CheckIsSameOperand, /*MI*/2, /*OpIdx*/1, /*OtherMI*/1, /*OtherOpIdx*/1,
        GIM_CheckConstantInt, /*MI*/2, /*Op*/2, -1,
        GIM_CheckIsSafeToFold, /*InsnID*/1,
        GIM_CheckIsSafeToFold, /*InsnID*/2,
        // (and:{ *:[i64] } (add:{ *:[i64] } GR64:{ *:[i64] }:$src, 1:{ *:[i64] }), (xor:{ *:[i64] } GR64:{ *:[i64] }:$src, -1:{ *:[i64] }))  =>  (BLCIC64rr:{ *:[i64] }:{ *:[i32] } GR64:{ *:[i64] }:$src)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::BLCIC64rr,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/1, /*OpIdx*/1, // src
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 18155,
        GIR_Done,
      // Label 270: @5305
      GIM_Try, /*On fail goto*//*Label 271*/ 5380, // Rule ID 18167 //
        GIM_CheckFeatures, GIFBS_HasTBM,
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_s64,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::GR64RegClassID,
        GIM_RecordInsn, /*DefineMI*/1, /*MI*/0, /*OpIdx*/1, // MIs[1]
        GIM_CheckOpcode, /*MI*/1, TargetOpcode::G_ADD,
        GIM_CheckType, /*MI*/1, /*Op*/1, /*Type*/GILLT_s64,
        GIM_CheckType, /*MI*/1, /*Op*/2, /*Type*/GILLT_s64,
        GIM_CheckRegBankForClass, /*MI*/1, /*Op*/1, /*RC*/X86::GR64RegClassID,
        GIM_CheckConstantInt, /*MI*/1, /*Op*/2, -1,
        GIM_RecordInsn, /*DefineMI*/2, /*MI*/0, /*OpIdx*/2, // MIs[2]
        GIM_CheckOpcode, /*MI*/2, TargetOpcode::G_XOR,
        GIM_CheckType, /*MI*/2, /*Op*/2, /*Type*/GILLT_s64,
        // MIs[2] src
        GIM_CheckIsSameOperand, /*MI*/2, /*OpIdx*/1, /*OtherMI*/1, /*OtherOpIdx*/1,
        GIM_CheckConstantInt, /*MI*/2, /*Op*/2, -1,
        GIM_CheckIsSafeToFold, /*InsnID*/1,
        GIM_CheckIsSafeToFold, /*InsnID*/2,
        // (and:{ *:[i64] } (add:{ *:[i64] } GR64:{ *:[i64] }:$src, -1:{ *:[i64] }), (xor:{ *:[i64] } GR64:{ *:[i64] }:$src, -1:{ *:[i64] }))  =>  (TZMSK64rr:{ *:[i64] }:{ *:[i32] } GR64:{ *:[i64] }:$src)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::TZMSK64rr,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/1, /*OpIdx*/1, // src
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 18167,
        GIR_Done,
      // Label 271: @5380
      GIM_Try, /*On fail goto*//*Label 272*/ 5455, // Rule ID 12228 //
        GIM_CheckFeatures, GIFBS_HasTBM,
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_s64,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::GR64RegClassID,
        GIM_RecordInsn, /*DefineMI*/1, /*MI*/0, /*OpIdx*/1, // MIs[1]
        GIM_CheckOpcode, /*MI*/1, TargetOpcode::G_XOR,
        GIM_CheckType, /*MI*/1, /*Op*/1, /*Type*/GILLT_s64,
        GIM_CheckType, /*MI*/1, /*Op*/2, /*Type*/GILLT_s64,
        GIM_CheckRegBankForClass, /*MI*/1, /*Op*/1, /*RC*/X86::GR64RegClassID,
        GIM_CheckConstantInt, /*MI*/1, /*Op*/2, -1,
        GIM_RecordInsn, /*DefineMI*/2, /*MI*/0, /*OpIdx*/2, // MIs[2]
        GIM_CheckOpcode, /*MI*/2, TargetOpcode::G_ADD,
        GIM_CheckType, /*MI*/2, /*Op*/2, /*Type*/GILLT_s64,
        // MIs[2] src
        GIM_CheckIsSameOperand, /*MI*/2, /*OpIdx*/1, /*OtherMI*/1, /*OtherOpIdx*/1,
        GIM_CheckConstantInt, /*MI*/2, /*Op*/2, 1,
        GIM_CheckIsSafeToFold, /*InsnID*/1,
        GIM_CheckIsSafeToFold, /*InsnID*/2,
        // (and:{ *:[i64] } (xor:{ *:[i64] } GR64:{ *:[i64] }:$src, -1:{ *:[i64] }), (add:{ *:[i64] } GR64:{ *:[i64] }:$src, 1:{ *:[i64] }))  =>  (BLCIC64rr:{ *:[i64] }:{ *:[i32] } GR64:{ *:[i64] }:$src)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::BLCIC64rr,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/1, /*OpIdx*/1, // src
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 12228,
        GIR_Done,
      // Label 272: @5455
      GIM_Try, /*On fail goto*//*Label 273*/ 5530, // Rule ID 12240 //
        GIM_CheckFeatures, GIFBS_HasTBM,
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_s64,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::GR64RegClassID,
        GIM_RecordInsn, /*DefineMI*/1, /*MI*/0, /*OpIdx*/1, // MIs[1]
        GIM_CheckOpcode, /*MI*/1, TargetOpcode::G_XOR,
        GIM_CheckType, /*MI*/1, /*Op*/1, /*Type*/GILLT_s64,
        GIM_CheckType, /*MI*/1, /*Op*/2, /*Type*/GILLT_s64,
        GIM_CheckRegBankForClass, /*MI*/1, /*Op*/1, /*RC*/X86::GR64RegClassID,
        GIM_CheckConstantInt, /*MI*/1, /*Op*/2, -1,
        GIM_RecordInsn, /*DefineMI*/2, /*MI*/0, /*OpIdx*/2, // MIs[2]
        GIM_CheckOpcode, /*MI*/2, TargetOpcode::G_ADD,
        GIM_CheckType, /*MI*/2, /*Op*/2, /*Type*/GILLT_s64,
        // MIs[2] src
        GIM_CheckIsSameOperand, /*MI*/2, /*OpIdx*/1, /*OtherMI*/1, /*OtherOpIdx*/1,
        GIM_CheckConstantInt, /*MI*/2, /*Op*/2, -1,
        GIM_CheckIsSafeToFold, /*InsnID*/1,
        GIM_CheckIsSafeToFold, /*InsnID*/2,
        // (and:{ *:[i64] } (xor:{ *:[i64] } GR64:{ *:[i64] }:$src, -1:{ *:[i64] }), (add:{ *:[i64] } GR64:{ *:[i64] }:$src, -1:{ *:[i64] }))  =>  (TZMSK64rr:{ *:[i64] }:{ *:[i32] } GR64:{ *:[i64] }:$src)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::TZMSK64rr,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/1, /*OpIdx*/1, // src
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 12240,
        GIR_Done,
      // Label 273: @5530
      GIM_Try, /*On fail goto*//*Label 274*/ 5584, // Rule ID 18149 //
        GIM_CheckFeatures, GIFBS_HasTBM,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::GR64RegClassID,
        GIM_RecordInsn, /*DefineMI*/1, /*MI*/0, /*OpIdx*/1, // MIs[1]
        GIM_CheckOpcode, /*MI*/1, TargetOpcode::G_ADD,
        GIM_CheckType, /*MI*/1, /*Op*/1, /*Type*/GILLT_s64,
        GIM_CheckType, /*MI*/1, /*Op*/2, /*Type*/GILLT_s64,
        GIM_CheckRegBankForClass, /*MI*/1, /*Op*/1, /*RC*/X86::GR64RegClassID,
        GIM_CheckConstantInt, /*MI*/1, /*Op*/2, 1,
        // MIs[0] src
        GIM_CheckIsSameOperand, /*MI*/0, /*OpIdx*/2, /*OtherMI*/1, /*OtherOpIdx*/1,
        GIM_CheckIsSafeToFold, /*InsnID*/1,
        // (and:{ *:[i64] } (add:{ *:[i64] } GR64:{ *:[i64] }:$src, 1:{ *:[i64] }), GR64:{ *:[i64] }:$src)  =>  (BLCFILL64rr:{ *:[i64] }:{ *:[i32] } GR64:{ *:[i64] }:$src)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::BLCFILL64rr,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/1, /*OpIdx*/1, // src
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 18149,
        GIR_Done,
      // Label 274: @5584
      GIM_Try, /*On fail goto*//*Label 275*/ 5638, // Rule ID 18675 //
        GIM_CheckFeatures, GIFBS_HasBMI,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::GR64RegClassID,
        GIM_RecordInsn, /*DefineMI*/1, /*MI*/0, /*OpIdx*/1, // MIs[1]
        GIM_CheckOpcode, /*MI*/1, TargetOpcode::G_ADD,
        GIM_CheckType, /*MI*/1, /*Op*/1, /*Type*/GILLT_s64,
        GIM_CheckType, /*MI*/1, /*Op*/2, /*Type*/GILLT_s64,
        GIM_CheckRegBankForClass, /*MI*/1, /*Op*/1, /*RC*/X86::GR64RegClassID,
        GIM_CheckConstantInt, /*MI*/1, /*Op*/2, -1,
        // MIs[0] src
        GIM_CheckIsSameOperand, /*MI*/0, /*OpIdx*/2, /*OtherMI*/1, /*OtherOpIdx*/1,
        GIM_CheckIsSafeToFold, /*InsnID*/1,
        // (and:{ *:[i64] } (add:{ *:[i64] } GR64:{ *:[i64] }:$src, -1:{ *:[i64] }), GR64:{ *:[i64] }:$src)  =>  (BLSR64rr:{ *:[i64] }:{ *:[i32] } GR64:{ *:[i64] }:$src)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::BLSR64rr,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/1, /*OpIdx*/1, // src
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 18675,
        GIR_Done,
      // Label 275: @5638
      GIM_Try, /*On fail goto*//*Label 276*/ 5692, // Rule ID 18679 //
        GIM_CheckFeatures, GIFBS_HasBMI,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::GR64RegClassID,
        GIM_RecordInsn, /*DefineMI*/1, /*MI*/0, /*OpIdx*/1, // MIs[1]
        GIM_CheckOpcode, /*MI*/1, TargetOpcode::G_SUB,
        GIM_CheckType, /*MI*/1, /*Op*/1, /*Type*/GILLT_s64,
        GIM_CheckType, /*MI*/1, /*Op*/2, /*Type*/GILLT_s64,
        GIM_CheckConstantInt, /*MI*/1, /*Op*/1, 0,
        GIM_CheckRegBankForClass, /*MI*/1, /*Op*/2, /*RC*/X86::GR64RegClassID,
        // MIs[0] src
        GIM_CheckIsSameOperand, /*MI*/0, /*OpIdx*/2, /*OtherMI*/1, /*OtherOpIdx*/2,
        GIM_CheckIsSafeToFold, /*InsnID*/1,
        // (and:{ *:[i64] } (sub:{ *:[i64] } 0:{ *:[i64] }, GR64:{ *:[i64] }:$src), GR64:{ *:[i64] }:$src)  =>  (BLSI64rr:{ *:[i64] }:{ *:[i32] } GR64:{ *:[i64] }:$src)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::BLSI64rr,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/1, /*OpIdx*/2, // src
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 18679,
        GIR_Done,
      // Label 276: @5692
      GIM_Try, /*On fail goto*//*Label 277*/ 5746, // Rule ID 12222 //
        GIM_CheckFeatures, GIFBS_HasTBM,
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_s64,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::GR64RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::GR64RegClassID,
        GIM_RecordInsn, /*DefineMI*/1, /*MI*/0, /*OpIdx*/2, // MIs[1]
        GIM_CheckOpcode, /*MI*/1, TargetOpcode::G_ADD,
        GIM_CheckType, /*MI*/1, /*Op*/2, /*Type*/GILLT_s64,
        // MIs[1] src
        GIM_CheckIsSameOperand, /*MI*/1, /*OpIdx*/1, /*OtherMI*/0, /*OtherOpIdx*/1,
        GIM_CheckConstantInt, /*MI*/1, /*Op*/2, 1,
        GIM_CheckIsSafeToFold, /*InsnID*/1,
        // (and:{ *:[i64] } GR64:{ *:[i64] }:$src, (add:{ *:[i64] } GR64:{ *:[i64] }:$src, 1:{ *:[i64] }))  =>  (BLCFILL64rr:{ *:[i64] }:{ *:[i32] } GR64:{ *:[i64] }:$src)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::BLCFILL64rr,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // src
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 12222,
        GIR_Done,
      // Label 277: @5746
      GIM_Try, /*On fail goto*//*Label 278*/ 5800, // Rule ID 16394 //
        GIM_CheckFeatures, GIFBS_HasBMI,
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_s64,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::GR64RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::GR64RegClassID,
        GIM_RecordInsn, /*DefineMI*/1, /*MI*/0, /*OpIdx*/2, // MIs[1]
        GIM_CheckOpcode, /*MI*/1, TargetOpcode::G_ADD,
        GIM_CheckType, /*MI*/1, /*Op*/2, /*Type*/GILLT_s64,
        // MIs[1] src
        GIM_CheckIsSameOperand, /*MI*/1, /*OpIdx*/1, /*OtherMI*/0, /*OtherOpIdx*/1,
        GIM_CheckConstantInt, /*MI*/1, /*Op*/2, -1,
        GIM_CheckIsSafeToFold, /*InsnID*/1,
        // (and:{ *:[i64] } GR64:{ *:[i64] }:$src, (add:{ *:[i64] } GR64:{ *:[i64] }:$src, -1:{ *:[i64] }))  =>  (BLSR64rr:{ *:[i64] }:{ *:[i32] } GR64:{ *:[i64] }:$src)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::BLSR64rr,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // src
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 16394,
        GIR_Done,
      // Label 278: @5800
      GIM_Try, /*On fail goto*//*Label 279*/ 5854, // Rule ID 16398 //
        GIM_CheckFeatures, GIFBS_HasBMI,
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_s64,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::GR64RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::GR64RegClassID,
        GIM_RecordInsn, /*DefineMI*/1, /*MI*/0, /*OpIdx*/2, // MIs[1]
        GIM_CheckOpcode, /*MI*/1, TargetOpcode::G_SUB,
        GIM_CheckType, /*MI*/1, /*Op*/1, /*Type*/GILLT_s64,
        GIM_CheckConstantInt, /*MI*/1, /*Op*/1, 0,
        // MIs[1] src
        GIM_CheckIsSameOperand, /*MI*/1, /*OpIdx*/2, /*OtherMI*/0, /*OtherOpIdx*/1,
        GIM_CheckIsSafeToFold, /*InsnID*/1,
        // (and:{ *:[i64] } GR64:{ *:[i64] }:$src, (sub:{ *:[i64] } 0:{ *:[i64] }, GR64:{ *:[i64] }:$src))  =>  (BLSI64rr:{ *:[i64] }:{ *:[i32] } GR64:{ *:[i64] }:$src)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::BLSI64rr,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // src
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 16398,
        GIR_Done,
      // Label 279: @5854
      GIM_Try, /*On fail goto*//*Label 280*/ 5899, // Rule ID 16199 //
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_s64,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::GR64RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::GR64RegClassID,
        GIM_RecordInsn, /*DefineMI*/1, /*MI*/0, /*OpIdx*/2, // MIs[1]
        GIM_CheckOpcode, /*MI*/1, TargetOpcode::G_CONSTANT,
        GIM_CheckI64ImmPredicate, /*MI*/1, /*Predicate*/GIPFP_I64_Predicate_i64immSExt8,
        // MIs[1] Operand 1
        // No operand predicates
        GIM_CheckIsSafeToFold, /*InsnID*/1,
        // (and:{ *:[i64] } GR64:{ *:[i64] }:$src1, (imm:{ *:[i64] })<<P:Predicate_i64immSExt8>>:$src2)  =>  (AND64ri8:{ *:[i64] }:{ *:[i32] } GR64:{ *:[i64] }:$src1, (imm:{ *:[i64] })<<P:Predicate_i64immSExt8>>:$src2)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::AND64ri8,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // src1
        GIR_CopyConstantAsSImm, /*NewInsnID*/0, /*OldInsnID*/1, // src2
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 16199,
        GIR_Done,
      // Label 280: @5899
      GIM_Try, /*On fail goto*//*Label 281*/ 5944, // Rule ID 16200 //
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_s64,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::GR64RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::GR64RegClassID,
        GIM_RecordInsn, /*DefineMI*/1, /*MI*/0, /*OpIdx*/2, // MIs[1]
        GIM_CheckOpcode, /*MI*/1, TargetOpcode::G_CONSTANT,
        GIM_CheckI64ImmPredicate, /*MI*/1, /*Predicate*/GIPFP_I64_Predicate_i64immSExt32,
        // MIs[1] Operand 1
        // No operand predicates
        GIM_CheckIsSafeToFold, /*InsnID*/1,
        // (and:{ *:[i64] } GR64:{ *:[i64] }:$src1, (imm:{ *:[i64] })<<P:Predicate_i64immSExt32>>:$src2)  =>  (AND64ri32:{ *:[i64] }:{ *:[i32] } GR64:{ *:[i64] }:$src1, (imm:{ *:[i64] })<<P:Predicate_i64immSExt32>>:$src2)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::AND64ri32,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // src1
        GIR_CopyConstantAsSImm, /*NewInsnID*/0, /*OldInsnID*/1, // src2
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 16200,
        GIR_Done,
      // Label 281: @5944
      GIM_Try, /*On fail goto*//*Label 282*/ 6005, // Rule ID 12250 //
        GIM_CheckFeatures, GIFBS_HasBMI,
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_s64,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::GR64RegClassID,
        GIM_RecordInsn, /*DefineMI*/1, /*MI*/0, /*OpIdx*/1, // MIs[1]
        GIM_CheckOpcode, /*MI*/1, TargetOpcode::G_XOR,
        GIM_CheckType, /*MI*/1, /*Op*/1, /*Type*/GILLT_s64,
        GIM_CheckType, /*MI*/1, /*Op*/2, /*Type*/GILLT_s64,
        GIM_CheckRegBankForClass, /*MI*/1, /*Op*/1, /*RC*/X86::GR64RegClassID,
        GIM_CheckConstantInt, /*MI*/1, /*Op*/2, -1,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::GR64RegClassID,
        GIM_CheckIsSafeToFold, /*InsnID*/1,
        // (and:{ *:[i64] } (xor:{ *:[i64] } GR64:{ *:[i64] }:$src1, -1:{ *:[i64] }), GR64:{ *:[i64] }:$src2)  =>  (ANDN64rr:{ *:[i64] }:{ *:[i32] } GR64:{ *:[i64] }:$src1, GR64:{ *:[i64] }:$src2)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::ANDN64rr,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/1, /*OpIdx*/1, // src1
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/2, // src2
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 12250,
        GIR_Done,
      // Label 282: @6005
      GIM_Try, /*On fail goto*//*Label 283*/ 6066, // Rule ID 18169 //
        GIM_CheckFeatures, GIFBS_HasBMI,
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_s64,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::GR64RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::GR64RegClassID,
        GIM_RecordInsn, /*DefineMI*/1, /*MI*/0, /*OpIdx*/2, // MIs[1]
        GIM_CheckOpcode, /*MI*/1, TargetOpcode::G_XOR,
        GIM_CheckType, /*MI*/1, /*Op*/1, /*Type*/GILLT_s64,
        GIM_CheckType, /*MI*/1, /*Op*/2, /*Type*/GILLT_s64,
        GIM_CheckRegBankForClass, /*MI*/1, /*Op*/1, /*RC*/X86::GR64RegClassID,
        GIM_CheckConstantInt, /*MI*/1, /*Op*/2, -1,
        GIM_CheckIsSafeToFold, /*InsnID*/1,
        // (and:{ *:[i64] } GR64:{ *:[i64] }:$src2, (xor:{ *:[i64] } GR64:{ *:[i64] }:$src1, -1:{ *:[i64] }))  =>  (ANDN64rr:{ *:[i64] }:{ *:[i32] } GR64:{ *:[i64] }:$src1, GR64:{ *:[i64] }:$src2)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::ANDN64rr,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/1, /*OpIdx*/1, // src1
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // src2
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 18169,
        GIR_Done,
      // Label 283: @6066
      GIM_Try, /*On fail goto*//*Label 284*/ 6094, // Rule ID 16189 //
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_s64,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::GR64RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::GR64RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::GR64RegClassID,
        // (and:{ *:[i64] } GR64:{ *:[i64] }:$src1, GR64:{ *:[i64] }:$src2)  =>  (AND64rr:{ *:[i64] }:{ *:[i32] } GR64:{ *:[i64] }:$src1, GR64:{ *:[i64] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::AND64rr,
        GIR_AddImplicitDef, /*InsnID*/0, X86::EFLAGS,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 16189,
        GIR_Done,
      // Label 284: @6094
      GIM_Reject,
    // Label 269: @6095
    GIM_Reject,
    // Label 228: @6096
    GIM_Try, /*On fail goto*//*Label 285*/ 6188, // Rule ID 13732 //
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v2s1,
      GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v2s1,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VK1RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VK2RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VK2RegClassID,
      // (and:{ *:[v2i1] } VK2:{ *:[v2i1] }:$src1, VK2:{ *:[v2i1] }:$src2)  =>  (COPY_TO_REGCLASS:{ *:[v2i1] } (KANDWrr:{ *:[v16i1] } (COPY_TO_REGCLASS:{ *:[v16i1] } VK2:{ *:[v2i1] }:$src1, VK16:{ *:[i32] }), (COPY_TO_REGCLASS:{ *:[v16i1] } VK2:{ *:[v2i1] }:$src2, VK16:{ *:[i32] })), VK1:{ *:[i32] })
      GIR_MakeTempReg, /*TempRegID*/0, /*TypeID*/GILLT_v16s1,
      GIR_MakeTempReg, /*TempRegID*/1, /*TypeID*/GILLT_v16s1,
      GIR_MakeTempReg, /*TempRegID*/2, /*TypeID*/GILLT_v16s1,
      GIR_BuildMI, /*InsnID*/3, /*Opcode*/TargetOpcode::COPY,
      GIR_AddTempRegister, /*InsnID*/3, /*TempRegID*/2, /*TempRegFlags*/RegState::Define,
      GIR_Copy, /*NewInsnID*/3, /*OldInsnID*/0, /*OpIdx*/2, // src2
      GIR_ConstrainSelectedInstOperands, /*InsnID*/3,
      GIR_BuildMI, /*InsnID*/2, /*Opcode*/TargetOpcode::COPY,
      GIR_AddTempRegister, /*InsnID*/2, /*TempRegID*/1, /*TempRegFlags*/RegState::Define,
      GIR_Copy, /*NewInsnID*/2, /*OldInsnID*/0, /*OpIdx*/1, // src1
      GIR_ConstrainSelectedInstOperands, /*InsnID*/2,
      GIR_BuildMI, /*InsnID*/1, /*Opcode*/X86::KANDWrr,
      GIR_AddTempRegister, /*InsnID*/1, /*TempRegID*/0, /*TempRegFlags*/RegState::Define,
      GIR_AddTempRegister, /*InsnID*/1, /*TempRegID*/1, /*TempRegFlags*/0,
      GIR_AddTempRegister, /*InsnID*/1, /*TempRegID*/2, /*TempRegFlags*/0,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/1,
      GIR_BuildMI, /*InsnID*/0, /*Opcode*/TargetOpcode::COPY,
      GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
      GIR_AddTempRegister, /*InsnID*/0, /*TempRegID*/0, /*TempRegFlags*/0,
      GIR_EraseFromParent, /*InsnID*/0,
      GIR_ConstrainOperandRC, /*InsnID*/0, /*Op*/0, /*RC VK1*/8,
      // GIR_Coverage, 13732,
      GIR_Done,
    // Label 285: @6188
    GIM_Reject,
    // Label 229: @6189
    GIM_Try, /*On fail goto*//*Label 286*/ 6269,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v2s64,
      GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v2s64,
      GIM_Try, /*On fail goto*//*Label 287*/ 6222, // Rule ID 1541 //
        GIM_CheckFeatures, GIFBS_HasAVX_NoVLX,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR128RegClassID,
        // (and:{ *:[v2i64] } VR128:{ *:[v2i64] }:$src1, VR128:{ *:[v2i64] }:$src2)  =>  (VPANDrr:{ *:[v2i64] } VR128:{ *:[v2i64] }:$src1, VR128:{ *:[v2i64] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VPANDrr,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 1541,
        GIR_Done,
      // Label 287: @6222
      GIM_Try, /*On fail goto*//*Label 288*/ 6245, // Rule ID 1543 //
        GIM_CheckFeatures, GIFBS_UseSSE2,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR128RegClassID,
        // (and:{ *:[v2i64] } VR128:{ *:[v2i64] }:$src1, VR128:{ *:[v2i64] }:$src2)  =>  (PANDrr:{ *:[v2i64] } VR128:{ *:[v2i64] }:$src1, VR128:{ *:[v2i64] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::PANDrr,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 1543,
        GIR_Done,
      // Label 288: @6245
      GIM_Try, /*On fail goto*//*Label 289*/ 6268, // Rule ID 4886 //
        GIM_CheckFeatures, GIFBS_HasAVX512_HasVLX,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128XRegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR128XRegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR128XRegClassID,
        // (and:{ *:[v2i64] } VR128X:{ *:[v2i64] }:$src1, VR128X:{ *:[v2i64] }:$src2)  =>  (VPANDQZ128rr:{ *:[v2i64] } VR128X:{ *:[v2i64] }:$src1, VR128X:{ *:[v2i64] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VPANDQZ128rr,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 4886,
        GIR_Done,
      // Label 289: @6268
      GIM_Reject,
    // Label 286: @6269
    GIM_Reject,
    // Label 230: @6270
    GIM_Try, /*On fail goto*//*Label 290*/ 6362, // Rule ID 13733 //
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v4s1,
      GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v4s1,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VK1RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VK4RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VK4RegClassID,
      // (and:{ *:[v4i1] } VK4:{ *:[v4i1] }:$src1, VK4:{ *:[v4i1] }:$src2)  =>  (COPY_TO_REGCLASS:{ *:[v4i1] } (KANDWrr:{ *:[v16i1] } (COPY_TO_REGCLASS:{ *:[v16i1] } VK4:{ *:[v4i1] }:$src1, VK16:{ *:[i32] }), (COPY_TO_REGCLASS:{ *:[v16i1] } VK4:{ *:[v4i1] }:$src2, VK16:{ *:[i32] })), VK1:{ *:[i32] })
      GIR_MakeTempReg, /*TempRegID*/0, /*TypeID*/GILLT_v16s1,
      GIR_MakeTempReg, /*TempRegID*/1, /*TypeID*/GILLT_v16s1,
      GIR_MakeTempReg, /*TempRegID*/2, /*TypeID*/GILLT_v16s1,
      GIR_BuildMI, /*InsnID*/3, /*Opcode*/TargetOpcode::COPY,
      GIR_AddTempRegister, /*InsnID*/3, /*TempRegID*/2, /*TempRegFlags*/RegState::Define,
      GIR_Copy, /*NewInsnID*/3, /*OldInsnID*/0, /*OpIdx*/2, // src2
      GIR_ConstrainSelectedInstOperands, /*InsnID*/3,
      GIR_BuildMI, /*InsnID*/2, /*Opcode*/TargetOpcode::COPY,
      GIR_AddTempRegister, /*InsnID*/2, /*TempRegID*/1, /*TempRegFlags*/RegState::Define,
      GIR_Copy, /*NewInsnID*/2, /*OldInsnID*/0, /*OpIdx*/1, // src1
      GIR_ConstrainSelectedInstOperands, /*InsnID*/2,
      GIR_BuildMI, /*InsnID*/1, /*Opcode*/X86::KANDWrr,
      GIR_AddTempRegister, /*InsnID*/1, /*TempRegID*/0, /*TempRegFlags*/RegState::Define,
      GIR_AddTempRegister, /*InsnID*/1, /*TempRegID*/1, /*TempRegFlags*/0,
      GIR_AddTempRegister, /*InsnID*/1, /*TempRegID*/2, /*TempRegFlags*/0,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/1,
      GIR_BuildMI, /*InsnID*/0, /*Opcode*/TargetOpcode::COPY,
      GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
      GIR_AddTempRegister, /*InsnID*/0, /*TempRegID*/0, /*TempRegFlags*/0,
      GIR_EraseFromParent, /*InsnID*/0,
      GIR_ConstrainOperandRC, /*InsnID*/0, /*Op*/0, /*RC VK1*/8,
      // GIR_Coverage, 13733,
      GIR_Done,
    // Label 290: @6362
    GIM_Reject,
    // Label 231: @6363
    GIM_Try, /*On fail goto*//*Label 291*/ 6443,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v4s64,
      GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v4s64,
      GIM_Try, /*On fail goto*//*Label 292*/ 6396, // Rule ID 1545 //
        GIM_CheckFeatures, GIFBS_HasAVX2_NoVLX,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR256RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR256RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR256RegClassID,
        // (and:{ *:[v4i64] } VR256:{ *:[v4i64] }:$src1, VR256:{ *:[v4i64] }:$src2)  =>  (VPANDYrr:{ *:[v4i64] } VR256:{ *:[v4i64] }:$src1, VR256:{ *:[v4i64] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VPANDYrr,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 1545,
        GIR_Done,
      // Label 292: @6396
      GIM_Try, /*On fail goto*//*Label 293*/ 6419, // Rule ID 4877 //
        GIM_CheckFeatures, GIFBS_HasAVX512_HasVLX,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR256XRegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR256XRegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR256XRegClassID,
        // (and:{ *:[v4i64] } VR256X:{ *:[v4i64] }:$src1, VR256X:{ *:[v4i64] }:$src2)  =>  (VPANDQZ256rr:{ *:[v4i64] } VR256X:{ *:[v4i64] }:$src1, VR256X:{ *:[v4i64] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VPANDQZ256rr,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 4877,
        GIR_Done,
      // Label 293: @6419
      GIM_Try, /*On fail goto*//*Label 294*/ 6442, // Rule ID 12515 //
        GIM_CheckFeatures, GIFBS_HasAVX1Only,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR256RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR256RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR256RegClassID,
        // (and:{ *:[v4i64] } VR256:{ *:[v4i64] }:$src1, VR256:{ *:[v4i64] }:$src2)  =>  (VANDPSYrr:{ *:[v4i64] } VR256:{ *:[v4i64] }:$src1, VR256:{ *:[v4i64] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VANDPSYrr,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 12515,
        GIR_Done,
      // Label 294: @6442
      GIM_Reject,
    // Label 291: @6443
    GIM_Reject,
    // Label 232: @6444
    GIM_Try, /*On fail goto*//*Label 295*/ 6552,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v8s1,
      GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v8s1,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VK8RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VK8RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VK8RegClassID,
      GIM_Try, /*On fail goto*//*Label 296*/ 6477, // Rule ID 3601 //
        GIM_CheckFeatures, GIFBS_HasDQI,
        // (and:{ *:[v8i1] } VK8:{ *:[v8i1] }:$src1, VK8:{ *:[v8i1] }:$src2)  =>  (KANDBrr:{ *:[v8i1] } VK8:{ *:[v8i1] }:$src1, VK8:{ *:[v8i1] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::KANDBrr,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 3601,
        GIR_Done,
      // Label 296: @6477
      GIM_Try, /*On fail goto*//*Label 297*/ 6551, // Rule ID 13730 //
        GIM_CheckFeatures, GIFBS_NoDQI,
        // (and:{ *:[v8i1] } VK8:{ *:[v8i1] }:$src1, VK8:{ *:[v8i1] }:$src2)  =>  (COPY_TO_REGCLASS:{ *:[v8i1] } (KANDWrr:{ *:[v16i1] } (COPY_TO_REGCLASS:{ *:[v16i1] } VK8:{ *:[v8i1] }:$src1, VK16:{ *:[i32] }), (COPY_TO_REGCLASS:{ *:[v16i1] } VK8:{ *:[v8i1] }:$src2, VK16:{ *:[i32] })), VK8:{ *:[i32] })
        GIR_MakeTempReg, /*TempRegID*/0, /*TypeID*/GILLT_v16s1,
        GIR_MakeTempReg, /*TempRegID*/1, /*TypeID*/GILLT_v16s1,
        GIR_MakeTempReg, /*TempRegID*/2, /*TypeID*/GILLT_v16s1,
        GIR_BuildMI, /*InsnID*/3, /*Opcode*/TargetOpcode::COPY,
        GIR_AddTempRegister, /*InsnID*/3, /*TempRegID*/2, /*TempRegFlags*/RegState::Define,
        GIR_Copy, /*NewInsnID*/3, /*OldInsnID*/0, /*OpIdx*/2, // src2
        GIR_ConstrainSelectedInstOperands, /*InsnID*/3,
        GIR_BuildMI, /*InsnID*/2, /*Opcode*/TargetOpcode::COPY,
        GIR_AddTempRegister, /*InsnID*/2, /*TempRegID*/1, /*TempRegFlags*/RegState::Define,
        GIR_Copy, /*NewInsnID*/2, /*OldInsnID*/0, /*OpIdx*/1, // src1
        GIR_ConstrainSelectedInstOperands, /*InsnID*/2,
        GIR_BuildMI, /*InsnID*/1, /*Opcode*/X86::KANDWrr,
        GIR_AddTempRegister, /*InsnID*/1, /*TempRegID*/0, /*TempRegFlags*/RegState::Define,
        GIR_AddTempRegister, /*InsnID*/1, /*TempRegID*/1, /*TempRegFlags*/0,
        GIR_AddTempRegister, /*InsnID*/1, /*TempRegID*/2, /*TempRegFlags*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/1,
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/TargetOpcode::COPY,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_AddTempRegister, /*InsnID*/0, /*TempRegID*/0, /*TempRegFlags*/0,
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainOperandRC, /*InsnID*/0, /*Op*/0, /*RC VK8*/12,
        // GIR_Coverage, 13730,
        GIR_Done,
      // Label 297: @6551
      GIM_Reject,
    // Label 295: @6552
    GIM_Reject,
    // Label 233: @6553
    GIM_Try, /*On fail goto*//*Label 298*/ 6584, // Rule ID 4868 //
      GIM_CheckFeatures, GIFBS_HasAVX512,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v8s64,
      GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v8s64,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR512RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR512RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR512RegClassID,
      // (and:{ *:[v8i64] } VR512:{ *:[v8i64] }:$src1, VR512:{ *:[v8i64] }:$src2)  =>  (VPANDQZrr:{ *:[v8i64] } VR512:{ *:[v8i64] }:$src1, VR512:{ *:[v8i64] }:$src2)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VPANDQZrr,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 4868,
      GIR_Done,
    // Label 298: @6584
    GIM_Reject,
    // Label 234: @6585
    GIM_Try, /*On fail goto*//*Label 299*/ 6616, // Rule ID 3602 //
      GIM_CheckFeatures, GIFBS_HasAVX512,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v16s1,
      GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v16s1,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VK16RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VK16RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VK16RegClassID,
      // (and:{ *:[v16i1] } VK16:{ *:[v16i1] }:$src1, VK16:{ *:[v16i1] }:$src2)  =>  (KANDWrr:{ *:[v16i1] } VK16:{ *:[v16i1] }:$src1, VK16:{ *:[v16i1] }:$src2)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::KANDWrr,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 3602,
      GIR_Done,
    // Label 299: @6616
    GIM_Reject,
    // Label 235: @6617
    GIM_Try, /*On fail goto*//*Label 300*/ 6648, // Rule ID 3603 //
      GIM_CheckFeatures, GIFBS_HasBWI,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v32s1,
      GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v32s1,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VK32RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VK32RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VK32RegClassID,
      // (and:{ *:[v32i1] } VK32:{ *:[v32i1] }:$src1, VK32:{ *:[v32i1] }:$src2)  =>  (KANDDrr:{ *:[v32i1] } VK32:{ *:[v32i1] }:$src1, VK32:{ *:[v32i1] }:$src2)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::KANDDrr,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 3603,
      GIR_Done,
    // Label 300: @6648
    GIM_Reject,
    // Label 236: @6649
    GIM_Try, /*On fail goto*//*Label 301*/ 6680, // Rule ID 3604 //
      GIM_CheckFeatures, GIFBS_HasBWI,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v64s1,
      GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v64s1,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VK64RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VK64RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VK64RegClassID,
      // (and:{ *:[v64i1] } VK64:{ *:[v64i1] }:$src1, VK64:{ *:[v64i1] }:$src2)  =>  (KANDQrr:{ *:[v64i1] } VK64:{ *:[v64i1] }:$src1, VK64:{ *:[v64i1] }:$src2)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::KANDQrr,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 3604,
      GIR_Done,
    // Label 301: @6680
    GIM_Reject,
    // Label 237: @6681
    GIM_Reject,
    // Label 4: @6682
    GIM_SwitchType, /*MI*/0, /*Op*/0, /*[*/0, 24, /*)*//*default:*//*Label 316*/ 9352,
    /*GILLT_s1*//*Label 302*/ 6712,
    /*GILLT_s8*//*Label 303*/ 6805,
    /*GILLT_s16*//*Label 304*/ 6871,
    /*GILLT_s32*//*Label 305*/ 6970,
    /*GILLT_s64*//*Label 306*/ 7867, 0, 0,
    /*GILLT_v2s1*//*Label 307*/ 8767,
    /*GILLT_v2s64*//*Label 308*/ 8860,
    /*GILLT_v4s1*//*Label 309*/ 8941, 0,
    /*GILLT_v4s64*//*Label 310*/ 9034,
    /*GILLT_v8s1*//*Label 311*/ 9115, 0, 0,
    /*GILLT_v8s64*//*Label 312*/ 9224,
    /*GILLT_v16s1*//*Label 313*/ 9256, 0, 0, 0,
    /*GILLT_v32s1*//*Label 314*/ 9288, 0, 0,
    /*GILLT_v64s1*//*Label 315*/ 9320,
    // Label 302: @6712
    GIM_Try, /*On fail goto*//*Label 317*/ 6804, // Rule ID 13739 //
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_s1,
      GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_s1,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VK1RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VK1RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VK1RegClassID,
      // (or:{ *:[v1i1] } VK1:{ *:[v1i1] }:$src1, VK1:{ *:[v1i1] }:$src2)  =>  (COPY_TO_REGCLASS:{ *:[v1i1] } (KORWrr:{ *:[v16i1] } (COPY_TO_REGCLASS:{ *:[v16i1] } VK1:{ *:[v1i1] }:$src1, VK16:{ *:[i32] }), (COPY_TO_REGCLASS:{ *:[v16i1] } VK1:{ *:[v1i1] }:$src2, VK16:{ *:[i32] })), VK1:{ *:[i32] })
      GIR_MakeTempReg, /*TempRegID*/0, /*TypeID*/GILLT_v16s1,
      GIR_MakeTempReg, /*TempRegID*/1, /*TypeID*/GILLT_v16s1,
      GIR_MakeTempReg, /*TempRegID*/2, /*TypeID*/GILLT_v16s1,
      GIR_BuildMI, /*InsnID*/3, /*Opcode*/TargetOpcode::COPY,
      GIR_AddTempRegister, /*InsnID*/3, /*TempRegID*/2, /*TempRegFlags*/RegState::Define,
      GIR_Copy, /*NewInsnID*/3, /*OldInsnID*/0, /*OpIdx*/2, // src2
      GIR_ConstrainSelectedInstOperands, /*InsnID*/3,
      GIR_BuildMI, /*InsnID*/2, /*Opcode*/TargetOpcode::COPY,
      GIR_AddTempRegister, /*InsnID*/2, /*TempRegID*/1, /*TempRegFlags*/RegState::Define,
      GIR_Copy, /*NewInsnID*/2, /*OldInsnID*/0, /*OpIdx*/1, // src1
      GIR_ConstrainSelectedInstOperands, /*InsnID*/2,
      GIR_BuildMI, /*InsnID*/1, /*Opcode*/X86::KORWrr,
      GIR_AddTempRegister, /*InsnID*/1, /*TempRegID*/0, /*TempRegFlags*/RegState::Define,
      GIR_AddTempRegister, /*InsnID*/1, /*TempRegID*/1, /*TempRegFlags*/0,
      GIR_AddTempRegister, /*InsnID*/1, /*TempRegID*/2, /*TempRegFlags*/0,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/1,
      GIR_BuildMI, /*InsnID*/0, /*Opcode*/TargetOpcode::COPY,
      GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
      GIR_AddTempRegister, /*InsnID*/0, /*TempRegID*/0, /*TempRegFlags*/0,
      GIR_EraseFromParent, /*InsnID*/0,
      GIR_ConstrainOperandRC, /*InsnID*/0, /*Op*/0, /*RC VK1*/8,
      // GIR_Coverage, 13739,
      GIR_Done,
    // Label 317: @6804
    GIM_Reject,
    // Label 303: @6805
    GIM_Try, /*On fail goto*//*Label 318*/ 6870,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_s8,
      GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_s8,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::GR8RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::GR8RegClassID,
      GIM_Try, /*On fail goto*//*Label 319*/ 6853, // Rule ID 16164 //
        GIM_RecordInsn, /*DefineMI*/1, /*MI*/0, /*OpIdx*/2, // MIs[1]
        GIM_CheckOpcode, /*MI*/1, TargetOpcode::G_CONSTANT,
        // MIs[1] Operand 1
        // No operand predicates
        GIM_CheckIsSafeToFold, /*InsnID*/1,
        // (or:{ *:[i8] } GR8:{ *:[i8] }:$src1, (imm:{ *:[i8] }):$src2)  =>  (OR8ri:{ *:[i8] }:{ *:[i32] } GR8:{ *:[i8] }:$src1, (imm:{ *:[i8] }):$src2)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::OR8ri,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // src1
        GIR_CopyConstantAsSImm, /*NewInsnID*/0, /*OldInsnID*/1, // src2
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 16164,
        GIR_Done,
      // Label 319: @6853
      GIM_Try, /*On fail goto*//*Label 320*/ 6869, // Rule ID 16156 //
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::GR8RegClassID,
        // (or:{ *:[i8] } GR8:{ *:[i8] }:$src1, GR8:{ *:[i8] }:$src2)  =>  (OR8rr:{ *:[i8] }:{ *:[i32] } GR8:{ *:[i8] }:$src1, GR8:{ *:[i8] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::OR8rr,
        GIR_AddImplicitDef, /*InsnID*/0, X86::EFLAGS,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 16156,
        GIR_Done,
      // Label 320: @6869
      GIM_Reject,
    // Label 318: @6870
    GIM_Reject,
    // Label 304: @6871
    GIM_Try, /*On fail goto*//*Label 321*/ 6969,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_s16,
      GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_s16,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::GR16RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::GR16RegClassID,
      GIM_Try, /*On fail goto*//*Label 322*/ 6922, // Rule ID 16167 //
        GIM_RecordInsn, /*DefineMI*/1, /*MI*/0, /*OpIdx*/2, // MIs[1]
        GIM_CheckOpcode, /*MI*/1, TargetOpcode::G_CONSTANT,
        GIM_CheckI64ImmPredicate, /*MI*/1, /*Predicate*/GIPFP_I64_Predicate_i16immSExt8,
        // MIs[1] Operand 1
        // No operand predicates
        GIM_CheckIsSafeToFold, /*InsnID*/1,
        // (or:{ *:[i16] } GR16:{ *:[i16] }:$src1, (imm:{ *:[i16] })<<P:Predicate_i16immSExt8>>:$src2)  =>  (OR16ri8:{ *:[i16] }:{ *:[i32] } GR16:{ *:[i16] }:$src1, (imm:{ *:[i16] })<<P:Predicate_i16immSExt8>>:$src2)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::OR16ri8,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // src1
        GIR_CopyConstantAsSImm, /*NewInsnID*/0, /*OldInsnID*/1, // src2
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 16167,
        GIR_Done,
      // Label 322: @6922
      GIM_Try, /*On fail goto*//*Label 323*/ 6952, // Rule ID 16165 //
        GIM_RecordInsn, /*DefineMI*/1, /*MI*/0, /*OpIdx*/2, // MIs[1]
        GIM_CheckOpcode, /*MI*/1, TargetOpcode::G_CONSTANT,
        // MIs[1] Operand 1
        // No operand predicates
        GIM_CheckIsSafeToFold, /*InsnID*/1,
        // (or:{ *:[i16] } GR16:{ *:[i16] }:$src1, (imm:{ *:[i16] }):$src2)  =>  (OR16ri:{ *:[i16] }:{ *:[i32] } GR16:{ *:[i16] }:$src1, (imm:{ *:[i16] }):$src2)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::OR16ri,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // src1
        GIR_CopyConstantAsSImm, /*NewInsnID*/0, /*OldInsnID*/1, // src2
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 16165,
        GIR_Done,
      // Label 323: @6952
      GIM_Try, /*On fail goto*//*Label 324*/ 6968, // Rule ID 16157 //
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::GR16RegClassID,
        // (or:{ *:[i16] } GR16:{ *:[i16] }:$src1, GR16:{ *:[i16] }:$src2)  =>  (OR16rr:{ *:[i16] }:{ *:[i32] } GR16:{ *:[i16] }:$src1, GR16:{ *:[i16] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::OR16rr,
        GIR_AddImplicitDef, /*InsnID*/0, X86::EFLAGS,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 16157,
        GIR_Done,
      // Label 324: @6968
      GIM_Reject,
    // Label 321: @6969
    GIM_Reject,
    // Label 305: @6970
    GIM_Try, /*On fail goto*//*Label 325*/ 7866,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_s32,
      GIM_Try, /*On fail goto*//*Label 326*/ 7051, // Rule ID 18162 //
        GIM_CheckFeatures, GIFBS_HasTBM,
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_s32,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::GR32RegClassID,
        GIM_RecordInsn, /*DefineMI*/1, /*MI*/0, /*OpIdx*/1, // MIs[1]
        GIM_CheckOpcode, /*MI*/1, TargetOpcode::G_ADD,
        GIM_CheckType, /*MI*/1, /*Op*/1, /*Type*/GILLT_s32,
        GIM_CheckType, /*MI*/1, /*Op*/2, /*Type*/GILLT_s32,
        GIM_CheckRegBankForClass, /*MI*/1, /*Op*/1, /*RC*/X86::GR32RegClassID,
        GIM_CheckConstantInt, /*MI*/1, /*Op*/2, -1,
        GIM_RecordInsn, /*DefineMI*/2, /*MI*/0, /*OpIdx*/2, // MIs[2]
        GIM_CheckOpcode, /*MI*/2, TargetOpcode::G_XOR,
        GIM_CheckType, /*MI*/2, /*Op*/2, /*Type*/GILLT_s32,
        // MIs[2] src
        GIM_CheckIsSameOperand, /*MI*/2, /*OpIdx*/1, /*OtherMI*/1, /*OtherOpIdx*/1,
        GIM_CheckConstantInt, /*MI*/2, /*Op*/2, -1,
        GIM_CheckIsSafeToFold, /*InsnID*/1,
        GIM_CheckIsSafeToFold, /*InsnID*/2,
        // (or:{ *:[i32] } (add:{ *:[i32] } GR32:{ *:[i32] }:$src, -1:{ *:[i32] }), (xor:{ *:[i32] } GR32:{ *:[i32] }:$src, -1:{ *:[i32] }))  =>  (BLSIC32rr:{ *:[i32] }:{ *:[i32] } GR32:{ *:[i32] }:$src)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::BLSIC32rr,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/1, /*OpIdx*/1, // src
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 18162,
        GIR_Done,
      // Label 326: @7051
      GIM_Try, /*On fail goto*//*Label 327*/ 7126, // Rule ID 18164 //
        GIM_CheckFeatures, GIFBS_HasTBM,
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_s32,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::GR32RegClassID,
        GIM_RecordInsn, /*DefineMI*/1, /*MI*/0, /*OpIdx*/1, // MIs[1]
        GIM_CheckOpcode, /*MI*/1, TargetOpcode::G_ADD,
        GIM_CheckType, /*MI*/1, /*Op*/1, /*Type*/GILLT_s32,
        GIM_CheckType, /*MI*/1, /*Op*/2, /*Type*/GILLT_s32,
        GIM_CheckRegBankForClass, /*MI*/1, /*Op*/1, /*RC*/X86::GR32RegClassID,
        GIM_CheckConstantInt, /*MI*/1, /*Op*/2, 1,
        GIM_RecordInsn, /*DefineMI*/2, /*MI*/0, /*OpIdx*/2, // MIs[2]
        GIM_CheckOpcode, /*MI*/2, TargetOpcode::G_XOR,
        GIM_CheckType, /*MI*/2, /*Op*/2, /*Type*/GILLT_s32,
        // MIs[2] src
        GIM_CheckIsSameOperand, /*MI*/2, /*OpIdx*/1, /*OtherMI*/1, /*OtherOpIdx*/1,
        GIM_CheckConstantInt, /*MI*/2, /*Op*/2, -1,
        GIM_CheckIsSafeToFold, /*InsnID*/1,
        GIM_CheckIsSafeToFold, /*InsnID*/2,
        // (or:{ *:[i32] } (add:{ *:[i32] } GR32:{ *:[i32] }:$src, 1:{ *:[i32] }), (xor:{ *:[i32] } GR32:{ *:[i32] }:$src, -1:{ *:[i32] }))  =>  (T1MSKC32rr:{ *:[i32] }:{ *:[i32] } GR32:{ *:[i32] }:$src)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::T1MSKC32rr,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/1, /*OpIdx*/1, // src
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 18164,
        GIR_Done,
      // Label 327: @7126
      GIM_Try, /*On fail goto*//*Label 328*/ 7201, // Rule ID 18150 //
        GIM_CheckFeatures, GIFBS_HasTBM,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::GR32RegClassID,
        GIM_RecordInsn, /*DefineMI*/1, /*MI*/0, /*OpIdx*/1, // MIs[1]
        GIM_CheckOpcode, /*MI*/1, TargetOpcode::G_XOR,
        GIM_CheckType, /*MI*/1, /*Op*/1, /*Type*/GILLT_s32,
        GIM_CheckType, /*MI*/1, /*Op*/2, /*Type*/GILLT_s32,
        GIM_RecordInsn, /*DefineMI*/2, /*MI*/1, /*OpIdx*/1, // MIs[2]
        GIM_CheckOpcode, /*MI*/2, TargetOpcode::G_ADD,
        GIM_CheckType, /*MI*/2, /*Op*/1, /*Type*/GILLT_s32,
        GIM_CheckType, /*MI*/2, /*Op*/2, /*Type*/GILLT_s32,
        GIM_CheckRegBankForClass, /*MI*/2, /*Op*/1, /*RC*/X86::GR32RegClassID,
        GIM_CheckConstantInt, /*MI*/2, /*Op*/2, 1,
        GIM_CheckConstantInt, /*MI*/1, /*Op*/2, -1,
        // MIs[0] src
        GIM_CheckIsSameOperand, /*MI*/0, /*OpIdx*/2, /*OtherMI*/2, /*OtherOpIdx*/1,
        GIM_CheckIsSafeToFold, /*InsnID*/1,
        GIM_CheckIsSafeToFold, /*InsnID*/2,
        // (or:{ *:[i32] } (xor:{ *:[i32] } (add:{ *:[i32] } GR32:{ *:[i32] }:$src, 1:{ *:[i32] }), -1:{ *:[i32] }), GR32:{ *:[i32] }:$src)  =>  (BLCI32rr:{ *:[i32] }:{ *:[i32] } GR32:{ *:[i32] }:$src)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::BLCI32rr,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/2, /*OpIdx*/1, // src
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 18150,
        GIR_Done,
      // Label 328: @7201
      GIM_Try, /*On fail goto*//*Label 329*/ 7276, // Rule ID 12235 //
        GIM_CheckFeatures, GIFBS_HasTBM,
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_s32,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::GR32RegClassID,
        GIM_RecordInsn, /*DefineMI*/1, /*MI*/0, /*OpIdx*/1, // MIs[1]
        GIM_CheckOpcode, /*MI*/1, TargetOpcode::G_XOR,
        GIM_CheckType, /*MI*/1, /*Op*/1, /*Type*/GILLT_s32,
        GIM_CheckType, /*MI*/1, /*Op*/2, /*Type*/GILLT_s32,
        GIM_CheckRegBankForClass, /*MI*/1, /*Op*/1, /*RC*/X86::GR32RegClassID,
        GIM_CheckConstantInt, /*MI*/1, /*Op*/2, -1,
        GIM_RecordInsn, /*DefineMI*/2, /*MI*/0, /*OpIdx*/2, // MIs[2]
        GIM_CheckOpcode, /*MI*/2, TargetOpcode::G_ADD,
        GIM_CheckType, /*MI*/2, /*Op*/2, /*Type*/GILLT_s32,
        // MIs[2] src
        GIM_CheckIsSameOperand, /*MI*/2, /*OpIdx*/1, /*OtherMI*/1, /*OtherOpIdx*/1,
        GIM_CheckConstantInt, /*MI*/2, /*Op*/2, -1,
        GIM_CheckIsSafeToFold, /*InsnID*/1,
        GIM_CheckIsSafeToFold, /*InsnID*/2,
        // (or:{ *:[i32] } (xor:{ *:[i32] } GR32:{ *:[i32] }:$src, -1:{ *:[i32] }), (add:{ *:[i32] } GR32:{ *:[i32] }:$src, -1:{ *:[i32] }))  =>  (BLSIC32rr:{ *:[i32] }:{ *:[i32] } GR32:{ *:[i32] }:$src)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::BLSIC32rr,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/1, /*OpIdx*/1, // src
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 12235,
        GIR_Done,
      // Label 329: @7276
      GIM_Try, /*On fail goto*//*Label 330*/ 7351, // Rule ID 12237 //
        GIM_CheckFeatures, GIFBS_HasTBM,
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_s32,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::GR32RegClassID,
        GIM_RecordInsn, /*DefineMI*/1, /*MI*/0, /*OpIdx*/1, // MIs[1]
        GIM_CheckOpcode, /*MI*/1, TargetOpcode::G_XOR,
        GIM_CheckType, /*MI*/1, /*Op*/1, /*Type*/GILLT_s32,
        GIM_CheckType, /*MI*/1, /*Op*/2, /*Type*/GILLT_s32,
        GIM_CheckRegBankForClass, /*MI*/1, /*Op*/1, /*RC*/X86::GR32RegClassID,
        GIM_CheckConstantInt, /*MI*/1, /*Op*/2, -1,
        GIM_RecordInsn, /*DefineMI*/2, /*MI*/0, /*OpIdx*/2, // MIs[2]
        GIM_CheckOpcode, /*MI*/2, TargetOpcode::G_ADD,
        GIM_CheckType, /*MI*/2, /*Op*/2, /*Type*/GILLT_s32,
        // MIs[2] src
        GIM_CheckIsSameOperand, /*MI*/2, /*OpIdx*/1, /*OtherMI*/1, /*OtherOpIdx*/1,
        GIM_CheckConstantInt, /*MI*/2, /*Op*/2, 1,
        GIM_CheckIsSafeToFold, /*InsnID*/1,
        GIM_CheckIsSafeToFold, /*InsnID*/2,
        // (or:{ *:[i32] } (xor:{ *:[i32] } GR32:{ *:[i32] }:$src, -1:{ *:[i32] }), (add:{ *:[i32] } GR32:{ *:[i32] }:$src, 1:{ *:[i32] }))  =>  (T1MSKC32rr:{ *:[i32] }:{ *:[i32] } GR32:{ *:[i32] }:$src)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::T1MSKC32rr,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/1, /*OpIdx*/1, // src
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 12237,
        GIR_Done,
      // Label 330: @7351
      GIM_Try, /*On fail goto*//*Label 331*/ 7426, // Rule ID 12223 //
        GIM_CheckFeatures, GIFBS_HasTBM,
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_s32,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::GR32RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::GR32RegClassID,
        GIM_RecordInsn, /*DefineMI*/1, /*MI*/0, /*OpIdx*/2, // MIs[1]
        GIM_CheckOpcode, /*MI*/1, TargetOpcode::G_XOR,
        GIM_CheckType, /*MI*/1, /*Op*/1, /*Type*/GILLT_s32,
        GIM_CheckType, /*MI*/1, /*Op*/2, /*Type*/GILLT_s32,
        GIM_RecordInsn, /*DefineMI*/2, /*MI*/1, /*OpIdx*/1, // MIs[2]
        GIM_CheckOpcode, /*MI*/2, TargetOpcode::G_ADD,
        GIM_CheckType, /*MI*/2, /*Op*/2, /*Type*/GILLT_s32,
        // MIs[2] src
        GIM_CheckIsSameOperand, /*MI*/2, /*OpIdx*/1, /*OtherMI*/0, /*OtherOpIdx*/1,
        GIM_CheckConstantInt, /*MI*/2, /*Op*/2, 1,
        GIM_CheckConstantInt, /*MI*/1, /*Op*/2, -1,
        GIM_CheckIsSafeToFold, /*InsnID*/1,
        GIM_CheckIsSafeToFold, /*InsnID*/2,
        // (or:{ *:[i32] } GR32:{ *:[i32] }:$src, (xor:{ *:[i32] } (add:{ *:[i32] } GR32:{ *:[i32] }:$src, 1:{ *:[i32] }), -1:{ *:[i32] }))  =>  (BLCI32rr:{ *:[i32] }:{ *:[i32] } GR32:{ *:[i32] }:$src)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::BLCI32rr,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // src
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 12223,
        GIR_Done,
      // Label 331: @7426
      GIM_Try, /*On fail goto*//*Label 332*/ 7480, // Rule ID 18158 //
        GIM_CheckFeatures, GIFBS_HasTBM,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::GR32RegClassID,
        GIM_RecordInsn, /*DefineMI*/1, /*MI*/0, /*OpIdx*/1, // MIs[1]
        GIM_CheckOpcode, /*MI*/1, TargetOpcode::G_ADD,
        GIM_CheckType, /*MI*/1, /*Op*/1, /*Type*/GILLT_s32,
        GIM_CheckType, /*MI*/1, /*Op*/2, /*Type*/GILLT_s32,
        GIM_CheckRegBankForClass, /*MI*/1, /*Op*/1, /*RC*/X86::GR32RegClassID,
        GIM_CheckConstantInt, /*MI*/1, /*Op*/2, 1,
        // MIs[0] src
        GIM_CheckIsSameOperand, /*MI*/0, /*OpIdx*/2, /*OtherMI*/1, /*OtherOpIdx*/1,
        GIM_CheckIsSafeToFold, /*InsnID*/1,
        // (or:{ *:[i32] } (add:{ *:[i32] } GR32:{ *:[i32] }:$src, 1:{ *:[i32] }), GR32:{ *:[i32] }:$src)  =>  (BLCS32rr:{ *:[i32] }:{ *:[i32] } GR32:{ *:[i32] }:$src)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::BLCS32rr,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/1, /*OpIdx*/1, // src
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 18158,
        GIR_Done,
      // Label 332: @7480
      GIM_Try, /*On fail goto*//*Label 333*/ 7534, // Rule ID 18160 //
        GIM_CheckFeatures, GIFBS_HasTBM,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::GR32RegClassID,
        GIM_RecordInsn, /*DefineMI*/1, /*MI*/0, /*OpIdx*/1, // MIs[1]
        GIM_CheckOpcode, /*MI*/1, TargetOpcode::G_ADD,
        GIM_CheckType, /*MI*/1, /*Op*/1, /*Type*/GILLT_s32,
        GIM_CheckType, /*MI*/1, /*Op*/2, /*Type*/GILLT_s32,
        GIM_CheckRegBankForClass, /*MI*/1, /*Op*/1, /*RC*/X86::GR32RegClassID,
        GIM_CheckConstantInt, /*MI*/1, /*Op*/2, -1,
        // MIs[0] src
        GIM_CheckIsSameOperand, /*MI*/0, /*OpIdx*/2, /*OtherMI*/1, /*OtherOpIdx*/1,
        GIM_CheckIsSafeToFold, /*InsnID*/1,
        // (or:{ *:[i32] } (add:{ *:[i32] } GR32:{ *:[i32] }:$src, -1:{ *:[i32] }), GR32:{ *:[i32] }:$src)  =>  (BLSFILL32rr:{ *:[i32] }:{ *:[i32] } GR32:{ *:[i32] }:$src)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::BLSFILL32rr,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/1, /*OpIdx*/1, // src
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 18160,
        GIR_Done,
      // Label 333: @7534
      GIM_Try, /*On fail goto*//*Label 334*/ 7588, // Rule ID 18152 //
        GIM_CheckFeatures, GIFBS_HasTBM,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::GR32RegClassID,
        GIM_RecordInsn, /*DefineMI*/1, /*MI*/0, /*OpIdx*/1, // MIs[1]
        GIM_CheckOpcode, /*MI*/1, TargetOpcode::G_SUB,
        GIM_CheckType, /*MI*/1, /*Op*/1, /*Type*/GILLT_s32,
        GIM_CheckType, /*MI*/1, /*Op*/2, /*Type*/GILLT_s32,
        GIM_CheckConstantInt, /*MI*/1, /*Op*/1, -2,
        GIM_CheckRegBankForClass, /*MI*/1, /*Op*/2, /*RC*/X86::GR32RegClassID,
        // MIs[0] src
        GIM_CheckIsSameOperand, /*MI*/0, /*OpIdx*/2, /*OtherMI*/1, /*OtherOpIdx*/2,
        GIM_CheckIsSafeToFold, /*InsnID*/1,
        // (or:{ *:[i32] } (sub:{ *:[i32] } -2:{ *:[i32] }, GR32:{ *:[i32] }:$src), GR32:{ *:[i32] }:$src)  =>  (BLCI32rr:{ *:[i32] }:{ *:[i32] } GR32:{ *:[i32] }:$src)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::BLCI32rr,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/1, /*OpIdx*/2, // src
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 18152,
        GIR_Done,
      // Label 334: @7588
      GIM_Try, /*On fail goto*//*Label 335*/ 7642, // Rule ID 12231 //
        GIM_CheckFeatures, GIFBS_HasTBM,
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_s32,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::GR32RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::GR32RegClassID,
        GIM_RecordInsn, /*DefineMI*/1, /*MI*/0, /*OpIdx*/2, // MIs[1]
        GIM_CheckOpcode, /*MI*/1, TargetOpcode::G_ADD,
        GIM_CheckType, /*MI*/1, /*Op*/2, /*Type*/GILLT_s32,
        // MIs[1] src
        GIM_CheckIsSameOperand, /*MI*/1, /*OpIdx*/1, /*OtherMI*/0, /*OtherOpIdx*/1,
        GIM_CheckConstantInt, /*MI*/1, /*Op*/2, 1,
        GIM_CheckIsSafeToFold, /*InsnID*/1,
        // (or:{ *:[i32] } GR32:{ *:[i32] }:$src, (add:{ *:[i32] } GR32:{ *:[i32] }:$src, 1:{ *:[i32] }))  =>  (BLCS32rr:{ *:[i32] }:{ *:[i32] } GR32:{ *:[i32] }:$src)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::BLCS32rr,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // src
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 12231,
        GIR_Done,
      // Label 335: @7642
      GIM_Try, /*On fail goto*//*Label 336*/ 7696, // Rule ID 12233 //
        GIM_CheckFeatures, GIFBS_HasTBM,
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_s32,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::GR32RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::GR32RegClassID,
        GIM_RecordInsn, /*DefineMI*/1, /*MI*/0, /*OpIdx*/2, // MIs[1]
        GIM_CheckOpcode, /*MI*/1, TargetOpcode::G_ADD,
        GIM_CheckType, /*MI*/1, /*Op*/2, /*Type*/GILLT_s32,
        // MIs[1] src
        GIM_CheckIsSameOperand, /*MI*/1, /*OpIdx*/1, /*OtherMI*/0, /*OtherOpIdx*/1,
        GIM_CheckConstantInt, /*MI*/1, /*Op*/2, -1,
        GIM_CheckIsSafeToFold, /*InsnID*/1,
        // (or:{ *:[i32] } GR32:{ *:[i32] }:$src, (add:{ *:[i32] } GR32:{ *:[i32] }:$src, -1:{ *:[i32] }))  =>  (BLSFILL32rr:{ *:[i32] }:{ *:[i32] } GR32:{ *:[i32] }:$src)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::BLSFILL32rr,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // src
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 12233,
        GIR_Done,
      // Label 336: @7696
      GIM_Try, /*On fail goto*//*Label 337*/ 7750, // Rule ID 12225 //
        GIM_CheckFeatures, GIFBS_HasTBM,
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_s32,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::GR32RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::GR32RegClassID,
        GIM_RecordInsn, /*DefineMI*/1, /*MI*/0, /*OpIdx*/2, // MIs[1]
        GIM_CheckOpcode, /*MI*/1, TargetOpcode::G_SUB,
        GIM_CheckType, /*MI*/1, /*Op*/1, /*Type*/GILLT_s32,
        GIM_CheckConstantInt, /*MI*/1, /*Op*/1, -2,
        // MIs[1] src
        GIM_CheckIsSameOperand, /*MI*/1, /*OpIdx*/2, /*OtherMI*/0, /*OtherOpIdx*/1,
        GIM_CheckIsSafeToFold, /*InsnID*/1,
        // (or:{ *:[i32] } GR32:{ *:[i32] }:$src, (sub:{ *:[i32] } -2:{ *:[i32] }, GR32:{ *:[i32] }:$src))  =>  (BLCI32rr:{ *:[i32] }:{ *:[i32] } GR32:{ *:[i32] }:$src)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::BLCI32rr,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // src
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 12225,
        GIR_Done,
      // Label 337: @7750
      GIM_Try, /*On fail goto*//*Label 338*/ 7795, // Rule ID 16168 //
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_s32,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::GR32RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::GR32RegClassID,
        GIM_RecordInsn, /*DefineMI*/1, /*MI*/0, /*OpIdx*/2, // MIs[1]
        GIM_CheckOpcode, /*MI*/1, TargetOpcode::G_CONSTANT,
        GIM_CheckI64ImmPredicate, /*MI*/1, /*Predicate*/GIPFP_I64_Predicate_i32immSExt8,
        // MIs[1] Operand 1
        // No operand predicates
        GIM_CheckIsSafeToFold, /*InsnID*/1,
        // (or:{ *:[i32] } GR32:{ *:[i32] }:$src1, (imm:{ *:[i32] })<<P:Predicate_i32immSExt8>>:$src2)  =>  (OR32ri8:{ *:[i32] }:{ *:[i32] } GR32:{ *:[i32] }:$src1, (imm:{ *:[i32] })<<P:Predicate_i32immSExt8>>:$src2)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::OR32ri8,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // src1
        GIR_CopyConstantAsSImm, /*NewInsnID*/0, /*OldInsnID*/1, // src2
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 16168,
        GIR_Done,
      // Label 338: @7795
      GIM_Try, /*On fail goto*//*Label 339*/ 7837, // Rule ID 16166 //
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_s32,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::GR32RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::GR32RegClassID,
        GIM_RecordInsn, /*DefineMI*/1, /*MI*/0, /*OpIdx*/2, // MIs[1]
        GIM_CheckOpcode, /*MI*/1, TargetOpcode::G_CONSTANT,
        // MIs[1] Operand 1
        // No operand predicates
        GIM_CheckIsSafeToFold, /*InsnID*/1,
        // (or:{ *:[i32] } GR32:{ *:[i32] }:$src1, (imm:{ *:[i32] }):$src2)  =>  (OR32ri:{ *:[i32] }:{ *:[i32] } GR32:{ *:[i32] }:$src1, (imm:{ *:[i32] }):$src2)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::OR32ri,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // src1
        GIR_CopyConstantAsSImm, /*NewInsnID*/0, /*OldInsnID*/1, // src2
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 16166,
        GIR_Done,
      // Label 339: @7837
      GIM_Try, /*On fail goto*//*Label 340*/ 7865, // Rule ID 16158 //
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_s32,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::GR32RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::GR32RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::GR32RegClassID,
        // (or:{ *:[i32] } GR32:{ *:[i32] }:$src1, GR32:{ *:[i32] }:$src2)  =>  (OR32rr:{ *:[i32] }:{ *:[i32] } GR32:{ *:[i32] }:$src1, GR32:{ *:[i32] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::OR32rr,
        GIR_AddImplicitDef, /*InsnID*/0, X86::EFLAGS,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 16158,
        GIR_Done,
      // Label 340: @7865
      GIM_Reject,
    // Label 325: @7866
    GIM_Reject,
    // Label 306: @7867
    GIM_Try, /*On fail goto*//*Label 341*/ 8766,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_s64,
      GIM_Try, /*On fail goto*//*Label 342*/ 7948, // Rule ID 18163 //
        GIM_CheckFeatures, GIFBS_HasTBM,
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_s64,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::GR64RegClassID,
        GIM_RecordInsn, /*DefineMI*/1, /*MI*/0, /*OpIdx*/1, // MIs[1]
        GIM_CheckOpcode, /*MI*/1, TargetOpcode::G_ADD,
        GIM_CheckType, /*MI*/1, /*Op*/1, /*Type*/GILLT_s64,
        GIM_CheckType, /*MI*/1, /*Op*/2, /*Type*/GILLT_s64,
        GIM_CheckRegBankForClass, /*MI*/1, /*Op*/1, /*RC*/X86::GR64RegClassID,
        GIM_CheckConstantInt, /*MI*/1, /*Op*/2, -1,
        GIM_RecordInsn, /*DefineMI*/2, /*MI*/0, /*OpIdx*/2, // MIs[2]
        GIM_CheckOpcode, /*MI*/2, TargetOpcode::G_XOR,
        GIM_CheckType, /*MI*/2, /*Op*/2, /*Type*/GILLT_s64,
        // MIs[2] src
        GIM_CheckIsSameOperand, /*MI*/2, /*OpIdx*/1, /*OtherMI*/1, /*OtherOpIdx*/1,
        GIM_CheckConstantInt, /*MI*/2, /*Op*/2, -1,
        GIM_CheckIsSafeToFold, /*InsnID*/1,
        GIM_CheckIsSafeToFold, /*InsnID*/2,
        // (or:{ *:[i64] } (add:{ *:[i64] } GR64:{ *:[i64] }:$src, -1:{ *:[i64] }), (xor:{ *:[i64] } GR64:{ *:[i64] }:$src, -1:{ *:[i64] }))  =>  (BLSIC64rr:{ *:[i64] }:{ *:[i32] } GR64:{ *:[i64] }:$src)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::BLSIC64rr,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/1, /*OpIdx*/1, // src
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 18163,
        GIR_Done,
      // Label 342: @7948
      GIM_Try, /*On fail goto*//*Label 343*/ 8023, // Rule ID 18165 //
        GIM_CheckFeatures, GIFBS_HasTBM,
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_s64,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::GR64RegClassID,
        GIM_RecordInsn, /*DefineMI*/1, /*MI*/0, /*OpIdx*/1, // MIs[1]
        GIM_CheckOpcode, /*MI*/1, TargetOpcode::G_ADD,
        GIM_CheckType, /*MI*/1, /*Op*/1, /*Type*/GILLT_s64,
        GIM_CheckType, /*MI*/1, /*Op*/2, /*Type*/GILLT_s64,
        GIM_CheckRegBankForClass, /*MI*/1, /*Op*/1, /*RC*/X86::GR64RegClassID,
        GIM_CheckConstantInt, /*MI*/1, /*Op*/2, 1,
        GIM_RecordInsn, /*DefineMI*/2, /*MI*/0, /*OpIdx*/2, // MIs[2]
        GIM_CheckOpcode, /*MI*/2, TargetOpcode::G_XOR,
        GIM_CheckType, /*MI*/2, /*Op*/2, /*Type*/GILLT_s64,
        // MIs[2] src
        GIM_CheckIsSameOperand, /*MI*/2, /*OpIdx*/1, /*OtherMI*/1, /*OtherOpIdx*/1,
        GIM_CheckConstantInt, /*MI*/2, /*Op*/2, -1,
        GIM_CheckIsSafeToFold, /*InsnID*/1,
        GIM_CheckIsSafeToFold, /*InsnID*/2,
        // (or:{ *:[i64] } (add:{ *:[i64] } GR64:{ *:[i64] }:$src, 1:{ *:[i64] }), (xor:{ *:[i64] } GR64:{ *:[i64] }:$src, -1:{ *:[i64] }))  =>  (T1MSKC64rr:{ *:[i64] }:{ *:[i32] } GR64:{ *:[i64] }:$src)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::T1MSKC64rr,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/1, /*OpIdx*/1, // src
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 18165,
        GIR_Done,
      // Label 343: @8023
      GIM_Try, /*On fail goto*//*Label 344*/ 8098, // Rule ID 18151 //
        GIM_CheckFeatures, GIFBS_HasTBM,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::GR64RegClassID,
        GIM_RecordInsn, /*DefineMI*/1, /*MI*/0, /*OpIdx*/1, // MIs[1]
        GIM_CheckOpcode, /*MI*/1, TargetOpcode::G_XOR,
        GIM_CheckType, /*MI*/1, /*Op*/1, /*Type*/GILLT_s64,
        GIM_CheckType, /*MI*/1, /*Op*/2, /*Type*/GILLT_s64,
        GIM_RecordInsn, /*DefineMI*/2, /*MI*/1, /*OpIdx*/1, // MIs[2]
        GIM_CheckOpcode, /*MI*/2, TargetOpcode::G_ADD,
        GIM_CheckType, /*MI*/2, /*Op*/1, /*Type*/GILLT_s64,
        GIM_CheckType, /*MI*/2, /*Op*/2, /*Type*/GILLT_s64,
        GIM_CheckRegBankForClass, /*MI*/2, /*Op*/1, /*RC*/X86::GR64RegClassID,
        GIM_CheckConstantInt, /*MI*/2, /*Op*/2, 1,
        GIM_CheckConstantInt, /*MI*/1, /*Op*/2, -1,
        // MIs[0] src
        GIM_CheckIsSameOperand, /*MI*/0, /*OpIdx*/2, /*OtherMI*/2, /*OtherOpIdx*/1,
        GIM_CheckIsSafeToFold, /*InsnID*/1,
        GIM_CheckIsSafeToFold, /*InsnID*/2,
        // (or:{ *:[i64] } (xor:{ *:[i64] } (add:{ *:[i64] } GR64:{ *:[i64] }:$src, 1:{ *:[i64] }), -1:{ *:[i64] }), GR64:{ *:[i64] }:$src)  =>  (BLCI64rr:{ *:[i64] }:{ *:[i32] } GR64:{ *:[i64] }:$src)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::BLCI64rr,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/2, /*OpIdx*/1, // src
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 18151,
        GIR_Done,
      // Label 344: @8098
      GIM_Try, /*On fail goto*//*Label 345*/ 8173, // Rule ID 12236 //
        GIM_CheckFeatures, GIFBS_HasTBM,
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_s64,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::GR64RegClassID,
        GIM_RecordInsn, /*DefineMI*/1, /*MI*/0, /*OpIdx*/1, // MIs[1]
        GIM_CheckOpcode, /*MI*/1, TargetOpcode::G_XOR,
        GIM_CheckType, /*MI*/1, /*Op*/1, /*Type*/GILLT_s64,
        GIM_CheckType, /*MI*/1, /*Op*/2, /*Type*/GILLT_s64,
        GIM_CheckRegBankForClass, /*MI*/1, /*Op*/1, /*RC*/X86::GR64RegClassID,
        GIM_CheckConstantInt, /*MI*/1, /*Op*/2, -1,
        GIM_RecordInsn, /*DefineMI*/2, /*MI*/0, /*OpIdx*/2, // MIs[2]
        GIM_CheckOpcode, /*MI*/2, TargetOpcode::G_ADD,
        GIM_CheckType, /*MI*/2, /*Op*/2, /*Type*/GILLT_s64,
        // MIs[2] src
        GIM_CheckIsSameOperand, /*MI*/2, /*OpIdx*/1, /*OtherMI*/1, /*OtherOpIdx*/1,
        GIM_CheckConstantInt, /*MI*/2, /*Op*/2, -1,
        GIM_CheckIsSafeToFold, /*InsnID*/1,
        GIM_CheckIsSafeToFold, /*InsnID*/2,
        // (or:{ *:[i64] } (xor:{ *:[i64] } GR64:{ *:[i64] }:$src, -1:{ *:[i64] }), (add:{ *:[i64] } GR64:{ *:[i64] }:$src, -1:{ *:[i64] }))  =>  (BLSIC64rr:{ *:[i64] }:{ *:[i32] } GR64:{ *:[i64] }:$src)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::BLSIC64rr,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/1, /*OpIdx*/1, // src
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 12236,
        GIR_Done,
      // Label 345: @8173
      GIM_Try, /*On fail goto*//*Label 346*/ 8248, // Rule ID 12238 //
        GIM_CheckFeatures, GIFBS_HasTBM,
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_s64,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::GR64RegClassID,
        GIM_RecordInsn, /*DefineMI*/1, /*MI*/0, /*OpIdx*/1, // MIs[1]
        GIM_CheckOpcode, /*MI*/1, TargetOpcode::G_XOR,
        GIM_CheckType, /*MI*/1, /*Op*/1, /*Type*/GILLT_s64,
        GIM_CheckType, /*MI*/1, /*Op*/2, /*Type*/GILLT_s64,
        GIM_CheckRegBankForClass, /*MI*/1, /*Op*/1, /*RC*/X86::GR64RegClassID,
        GIM_CheckConstantInt, /*MI*/1, /*Op*/2, -1,
        GIM_RecordInsn, /*DefineMI*/2, /*MI*/0, /*OpIdx*/2, // MIs[2]
        GIM_CheckOpcode, /*MI*/2, TargetOpcode::G_ADD,
        GIM_CheckType, /*MI*/2, /*Op*/2, /*Type*/GILLT_s64,
        // MIs[2] src
        GIM_CheckIsSameOperand, /*MI*/2, /*OpIdx*/1, /*OtherMI*/1, /*OtherOpIdx*/1,
        GIM_CheckConstantInt, /*MI*/2, /*Op*/2, 1,
        GIM_CheckIsSafeToFold, /*InsnID*/1,
        GIM_CheckIsSafeToFold, /*InsnID*/2,
        // (or:{ *:[i64] } (xor:{ *:[i64] } GR64:{ *:[i64] }:$src, -1:{ *:[i64] }), (add:{ *:[i64] } GR64:{ *:[i64] }:$src, 1:{ *:[i64] }))  =>  (T1MSKC64rr:{ *:[i64] }:{ *:[i32] } GR64:{ *:[i64] }:$src)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::T1MSKC64rr,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/1, /*OpIdx*/1, // src
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 12238,
        GIR_Done,
      // Label 346: @8248
      GIM_Try, /*On fail goto*//*Label 347*/ 8323, // Rule ID 12224 //
        GIM_CheckFeatures, GIFBS_HasTBM,
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_s64,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::GR64RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::GR64RegClassID,
        GIM_RecordInsn, /*DefineMI*/1, /*MI*/0, /*OpIdx*/2, // MIs[1]
        GIM_CheckOpcode, /*MI*/1, TargetOpcode::G_XOR,
        GIM_CheckType, /*MI*/1, /*Op*/1, /*Type*/GILLT_s64,
        GIM_CheckType, /*MI*/1, /*Op*/2, /*Type*/GILLT_s64,
        GIM_RecordInsn, /*DefineMI*/2, /*MI*/1, /*OpIdx*/1, // MIs[2]
        GIM_CheckOpcode, /*MI*/2, TargetOpcode::G_ADD,
        GIM_CheckType, /*MI*/2, /*Op*/2, /*Type*/GILLT_s64,
        // MIs[2] src
        GIM_CheckIsSameOperand, /*MI*/2, /*OpIdx*/1, /*OtherMI*/0, /*OtherOpIdx*/1,
        GIM_CheckConstantInt, /*MI*/2, /*Op*/2, 1,
        GIM_CheckConstantInt, /*MI*/1, /*Op*/2, -1,
        GIM_CheckIsSafeToFold, /*InsnID*/1,
        GIM_CheckIsSafeToFold, /*InsnID*/2,
        // (or:{ *:[i64] } GR64:{ *:[i64] }:$src, (xor:{ *:[i64] } (add:{ *:[i64] } GR64:{ *:[i64] }:$src, 1:{ *:[i64] }), -1:{ *:[i64] }))  =>  (BLCI64rr:{ *:[i64] }:{ *:[i32] } GR64:{ *:[i64] }:$src)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::BLCI64rr,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // src
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 12224,
        GIR_Done,
      // Label 347: @8323
      GIM_Try, /*On fail goto*//*Label 348*/ 8377, // Rule ID 18159 //
        GIM_CheckFeatures, GIFBS_HasTBM,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::GR64RegClassID,
        GIM_RecordInsn, /*DefineMI*/1, /*MI*/0, /*OpIdx*/1, // MIs[1]
        GIM_CheckOpcode, /*MI*/1, TargetOpcode::G_ADD,
        GIM_CheckType, /*MI*/1, /*Op*/1, /*Type*/GILLT_s64,
        GIM_CheckType, /*MI*/1, /*Op*/2, /*Type*/GILLT_s64,
        GIM_CheckRegBankForClass, /*MI*/1, /*Op*/1, /*RC*/X86::GR64RegClassID,
        GIM_CheckConstantInt, /*MI*/1, /*Op*/2, 1,
        // MIs[0] src
        GIM_CheckIsSameOperand, /*MI*/0, /*OpIdx*/2, /*OtherMI*/1, /*OtherOpIdx*/1,
        GIM_CheckIsSafeToFold, /*InsnID*/1,
        // (or:{ *:[i64] } (add:{ *:[i64] } GR64:{ *:[i64] }:$src, 1:{ *:[i64] }), GR64:{ *:[i64] }:$src)  =>  (BLCS64rr:{ *:[i64] }:{ *:[i32] } GR64:{ *:[i64] }:$src)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::BLCS64rr,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/1, /*OpIdx*/1, // src
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 18159,
        GIR_Done,
      // Label 348: @8377
      GIM_Try, /*On fail goto*//*Label 349*/ 8431, // Rule ID 18161 //
        GIM_CheckFeatures, GIFBS_HasTBM,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::GR64RegClassID,
        GIM_RecordInsn, /*DefineMI*/1, /*MI*/0, /*OpIdx*/1, // MIs[1]
        GIM_CheckOpcode, /*MI*/1, TargetOpcode::G_ADD,
        GIM_CheckType, /*MI*/1, /*Op*/1, /*Type*/GILLT_s64,
        GIM_CheckType, /*MI*/1, /*Op*/2, /*Type*/GILLT_s64,
        GIM_CheckRegBankForClass, /*MI*/1, /*Op*/1, /*RC*/X86::GR64RegClassID,
        GIM_CheckConstantInt, /*MI*/1, /*Op*/2, -1,
        // MIs[0] src
        GIM_CheckIsSameOperand, /*MI*/0, /*OpIdx*/2, /*OtherMI*/1, /*OtherOpIdx*/1,
        GIM_CheckIsSafeToFold, /*InsnID*/1,
        // (or:{ *:[i64] } (add:{ *:[i64] } GR64:{ *:[i64] }:$src, -1:{ *:[i64] }), GR64:{ *:[i64] }:$src)  =>  (BLSFILL64rr:{ *:[i64] }:{ *:[i32] } GR64:{ *:[i64] }:$src)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::BLSFILL64rr,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/1, /*OpIdx*/1, // src
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 18161,
        GIR_Done,
      // Label 349: @8431
      GIM_Try, /*On fail goto*//*Label 350*/ 8485, // Rule ID 18153 //
        GIM_CheckFeatures, GIFBS_HasTBM,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::GR64RegClassID,
        GIM_RecordInsn, /*DefineMI*/1, /*MI*/0, /*OpIdx*/1, // MIs[1]
        GIM_CheckOpcode, /*MI*/1, TargetOpcode::G_SUB,
        GIM_CheckType, /*MI*/1, /*Op*/1, /*Type*/GILLT_s64,
        GIM_CheckType, /*MI*/1, /*Op*/2, /*Type*/GILLT_s64,
        GIM_CheckConstantInt, /*MI*/1, /*Op*/1, -2,
        GIM_CheckRegBankForClass, /*MI*/1, /*Op*/2, /*RC*/X86::GR64RegClassID,
        // MIs[0] src
        GIM_CheckIsSameOperand, /*MI*/0, /*OpIdx*/2, /*OtherMI*/1, /*OtherOpIdx*/2,
        GIM_CheckIsSafeToFold, /*InsnID*/1,
        // (or:{ *:[i64] } (sub:{ *:[i64] } -2:{ *:[i64] }, GR64:{ *:[i64] }:$src), GR64:{ *:[i64] }:$src)  =>  (BLCI64rr:{ *:[i64] }:{ *:[i32] } GR64:{ *:[i64] }:$src)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::BLCI64rr,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/1, /*OpIdx*/2, // src
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 18153,
        GIR_Done,
      // Label 350: @8485
      GIM_Try, /*On fail goto*//*Label 351*/ 8539, // Rule ID 12232 //
        GIM_CheckFeatures, GIFBS_HasTBM,
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_s64,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::GR64RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::GR64RegClassID,
        GIM_RecordInsn, /*DefineMI*/1, /*MI*/0, /*OpIdx*/2, // MIs[1]
        GIM_CheckOpcode, /*MI*/1, TargetOpcode::G_ADD,
        GIM_CheckType, /*MI*/1, /*Op*/2, /*Type*/GILLT_s64,
        // MIs[1] src
        GIM_CheckIsSameOperand, /*MI*/1, /*OpIdx*/1, /*OtherMI*/0, /*OtherOpIdx*/1,
        GIM_CheckConstantInt, /*MI*/1, /*Op*/2, 1,
        GIM_CheckIsSafeToFold, /*InsnID*/1,
        // (or:{ *:[i64] } GR64:{ *:[i64] }:$src, (add:{ *:[i64] } GR64:{ *:[i64] }:$src, 1:{ *:[i64] }))  =>  (BLCS64rr:{ *:[i64] }:{ *:[i32] } GR64:{ *:[i64] }:$src)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::BLCS64rr,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // src
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 12232,
        GIR_Done,
      // Label 351: @8539
      GIM_Try, /*On fail goto*//*Label 352*/ 8593, // Rule ID 12234 //
        GIM_CheckFeatures, GIFBS_HasTBM,
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_s64,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::GR64RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::GR64RegClassID,
        GIM_RecordInsn, /*DefineMI*/1, /*MI*/0, /*OpIdx*/2, // MIs[1]
        GIM_CheckOpcode, /*MI*/1, TargetOpcode::G_ADD,
        GIM_CheckType, /*MI*/1, /*Op*/2, /*Type*/GILLT_s64,
        // MIs[1] src
        GIM_CheckIsSameOperand, /*MI*/1, /*OpIdx*/1, /*OtherMI*/0, /*OtherOpIdx*/1,
        GIM_CheckConstantInt, /*MI*/1, /*Op*/2, -1,
        GIM_CheckIsSafeToFold, /*InsnID*/1,
        // (or:{ *:[i64] } GR64:{ *:[i64] }:$src, (add:{ *:[i64] } GR64:{ *:[i64] }:$src, -1:{ *:[i64] }))  =>  (BLSFILL64rr:{ *:[i64] }:{ *:[i32] } GR64:{ *:[i64] }:$src)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::BLSFILL64rr,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // src
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 12234,
        GIR_Done,
      // Label 352: @8593
      GIM_Try, /*On fail goto*//*Label 353*/ 8647, // Rule ID 12226 //
        GIM_CheckFeatures, GIFBS_HasTBM,
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_s64,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::GR64RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::GR64RegClassID,
        GIM_RecordInsn, /*DefineMI*/1, /*MI*/0, /*OpIdx*/2, // MIs[1]
        GIM_CheckOpcode, /*MI*/1, TargetOpcode::G_SUB,
        GIM_CheckType, /*MI*/1, /*Op*/1, /*Type*/GILLT_s64,
        GIM_CheckConstantInt, /*MI*/1, /*Op*/1, -2,
        // MIs[1] src
        GIM_CheckIsSameOperand, /*MI*/1, /*OpIdx*/2, /*OtherMI*/0, /*OtherOpIdx*/1,
        GIM_CheckIsSafeToFold, /*InsnID*/1,
        // (or:{ *:[i64] } GR64:{ *:[i64] }:$src, (sub:{ *:[i64] } -2:{ *:[i64] }, GR64:{ *:[i64] }:$src))  =>  (BLCI64rr:{ *:[i64] }:{ *:[i32] } GR64:{ *:[i64] }:$src)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::BLCI64rr,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // src
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 12226,
        GIR_Done,
      // Label 353: @8647
      GIM_Try, /*On fail goto*//*Label 354*/ 8692, // Rule ID 16169 //
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_s64,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::GR64RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::GR64RegClassID,
        GIM_RecordInsn, /*DefineMI*/1, /*MI*/0, /*OpIdx*/2, // MIs[1]
        GIM_CheckOpcode, /*MI*/1, TargetOpcode::G_CONSTANT,
        GIM_CheckI64ImmPredicate, /*MI*/1, /*Predicate*/GIPFP_I64_Predicate_i64immSExt8,
        // MIs[1] Operand 1
        // No operand predicates
        GIM_CheckIsSafeToFold, /*InsnID*/1,
        // (or:{ *:[i64] } GR64:{ *:[i64] }:$src1, (imm:{ *:[i64] })<<P:Predicate_i64immSExt8>>:$src2)  =>  (OR64ri8:{ *:[i64] }:{ *:[i32] } GR64:{ *:[i64] }:$src1, (imm:{ *:[i64] })<<P:Predicate_i64immSExt8>>:$src2)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::OR64ri8,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // src1
        GIR_CopyConstantAsSImm, /*NewInsnID*/0, /*OldInsnID*/1, // src2
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 16169,
        GIR_Done,
      // Label 354: @8692
      GIM_Try, /*On fail goto*//*Label 355*/ 8737, // Rule ID 16170 //
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_s64,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::GR64RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::GR64RegClassID,
        GIM_RecordInsn, /*DefineMI*/1, /*MI*/0, /*OpIdx*/2, // MIs[1]
        GIM_CheckOpcode, /*MI*/1, TargetOpcode::G_CONSTANT,
        GIM_CheckI64ImmPredicate, /*MI*/1, /*Predicate*/GIPFP_I64_Predicate_i64immSExt32,
        // MIs[1] Operand 1
        // No operand predicates
        GIM_CheckIsSafeToFold, /*InsnID*/1,
        // (or:{ *:[i64] } GR64:{ *:[i64] }:$src1, (imm:{ *:[i64] })<<P:Predicate_i64immSExt32>>:$src2)  =>  (OR64ri32:{ *:[i64] }:{ *:[i32] } GR64:{ *:[i64] }:$src1, (imm:{ *:[i64] })<<P:Predicate_i64immSExt32>>:$src2)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::OR64ri32,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // src1
        GIR_CopyConstantAsSImm, /*NewInsnID*/0, /*OldInsnID*/1, // src2
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 16170,
        GIR_Done,
      // Label 355: @8737
      GIM_Try, /*On fail goto*//*Label 356*/ 8765, // Rule ID 16159 //
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_s64,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::GR64RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::GR64RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::GR64RegClassID,
        // (or:{ *:[i64] } GR64:{ *:[i64] }:$src1, GR64:{ *:[i64] }:$src2)  =>  (OR64rr:{ *:[i64] }:{ *:[i32] } GR64:{ *:[i64] }:$src1, GR64:{ *:[i64] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::OR64rr,
        GIR_AddImplicitDef, /*InsnID*/0, X86::EFLAGS,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 16159,
        GIR_Done,
      // Label 356: @8765
      GIM_Reject,
    // Label 341: @8766
    GIM_Reject,
    // Label 307: @8767
    GIM_Try, /*On fail goto*//*Label 357*/ 8859, // Rule ID 13740 //
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v2s1,
      GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v2s1,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VK1RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VK2RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VK2RegClassID,
      // (or:{ *:[v2i1] } VK2:{ *:[v2i1] }:$src1, VK2:{ *:[v2i1] }:$src2)  =>  (COPY_TO_REGCLASS:{ *:[v2i1] } (KORWrr:{ *:[v16i1] } (COPY_TO_REGCLASS:{ *:[v16i1] } VK2:{ *:[v2i1] }:$src1, VK16:{ *:[i32] }), (COPY_TO_REGCLASS:{ *:[v16i1] } VK2:{ *:[v2i1] }:$src2, VK16:{ *:[i32] })), VK1:{ *:[i32] })
      GIR_MakeTempReg, /*TempRegID*/0, /*TypeID*/GILLT_v16s1,
      GIR_MakeTempReg, /*TempRegID*/1, /*TypeID*/GILLT_v16s1,
      GIR_MakeTempReg, /*TempRegID*/2, /*TypeID*/GILLT_v16s1,
      GIR_BuildMI, /*InsnID*/3, /*Opcode*/TargetOpcode::COPY,
      GIR_AddTempRegister, /*InsnID*/3, /*TempRegID*/2, /*TempRegFlags*/RegState::Define,
      GIR_Copy, /*NewInsnID*/3, /*OldInsnID*/0, /*OpIdx*/2, // src2
      GIR_ConstrainSelectedInstOperands, /*InsnID*/3,
      GIR_BuildMI, /*InsnID*/2, /*Opcode*/TargetOpcode::COPY,
      GIR_AddTempRegister, /*InsnID*/2, /*TempRegID*/1, /*TempRegFlags*/RegState::Define,
      GIR_Copy, /*NewInsnID*/2, /*OldInsnID*/0, /*OpIdx*/1, // src1
      GIR_ConstrainSelectedInstOperands, /*InsnID*/2,
      GIR_BuildMI, /*InsnID*/1, /*Opcode*/X86::KORWrr,
      GIR_AddTempRegister, /*InsnID*/1, /*TempRegID*/0, /*TempRegFlags*/RegState::Define,
      GIR_AddTempRegister, /*InsnID*/1, /*TempRegID*/1, /*TempRegFlags*/0,
      GIR_AddTempRegister, /*InsnID*/1, /*TempRegID*/2, /*TempRegFlags*/0,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/1,
      GIR_BuildMI, /*InsnID*/0, /*Opcode*/TargetOpcode::COPY,
      GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
      GIR_AddTempRegister, /*InsnID*/0, /*TempRegID*/0, /*TempRegFlags*/0,
      GIR_EraseFromParent, /*InsnID*/0,
      GIR_ConstrainOperandRC, /*InsnID*/0, /*Op*/0, /*RC VK1*/8,
      // GIR_Coverage, 13740,
      GIR_Done,
    // Label 357: @8859
    GIM_Reject,
    // Label 308: @8860
    GIM_Try, /*On fail goto*//*Label 358*/ 8940,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v2s64,
      GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v2s64,
      GIM_Try, /*On fail goto*//*Label 359*/ 8893, // Rule ID 1547 //
        GIM_CheckFeatures, GIFBS_HasAVX_NoVLX,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR128RegClassID,
        // (or:{ *:[v2i64] } VR128:{ *:[v2i64] }:$src1, VR128:{ *:[v2i64] }:$src2)  =>  (VPORrr:{ *:[v2i64] } VR128:{ *:[v2i64] }:$src1, VR128:{ *:[v2i64] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VPORrr,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 1547,
        GIR_Done,
      // Label 359: @8893
      GIM_Try, /*On fail goto*//*Label 360*/ 8916, // Rule ID 1549 //
        GIM_CheckFeatures, GIFBS_UseSSE2,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR128RegClassID,
        // (or:{ *:[v2i64] } VR128:{ *:[v2i64] }:$src1, VR128:{ *:[v2i64] }:$src2)  =>  (PORrr:{ *:[v2i64] } VR128:{ *:[v2i64] }:$src1, VR128:{ *:[v2i64] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::PORrr,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 1549,
        GIR_Done,
      // Label 360: @8916
      GIM_Try, /*On fail goto*//*Label 361*/ 8939, // Rule ID 4934 //
        GIM_CheckFeatures, GIFBS_HasAVX512_HasVLX,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128XRegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR128XRegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR128XRegClassID,
        // (or:{ *:[v2i64] } VR128X:{ *:[v2i64] }:$src1, VR128X:{ *:[v2i64] }:$src2)  =>  (VPORQZ128rr:{ *:[v2i64] } VR128X:{ *:[v2i64] }:$src1, VR128X:{ *:[v2i64] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VPORQZ128rr,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 4934,
        GIR_Done,
      // Label 361: @8939
      GIM_Reject,
    // Label 358: @8940
    GIM_Reject,
    // Label 309: @8941
    GIM_Try, /*On fail goto*//*Label 362*/ 9033, // Rule ID 13741 //
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v4s1,
      GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v4s1,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VK1RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VK4RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VK4RegClassID,
      // (or:{ *:[v4i1] } VK4:{ *:[v4i1] }:$src1, VK4:{ *:[v4i1] }:$src2)  =>  (COPY_TO_REGCLASS:{ *:[v4i1] } (KORWrr:{ *:[v16i1] } (COPY_TO_REGCLASS:{ *:[v16i1] } VK4:{ *:[v4i1] }:$src1, VK16:{ *:[i32] }), (COPY_TO_REGCLASS:{ *:[v16i1] } VK4:{ *:[v4i1] }:$src2, VK16:{ *:[i32] })), VK1:{ *:[i32] })
      GIR_MakeTempReg, /*TempRegID*/0, /*TypeID*/GILLT_v16s1,
      GIR_MakeTempReg, /*TempRegID*/1, /*TypeID*/GILLT_v16s1,
      GIR_MakeTempReg, /*TempRegID*/2, /*TypeID*/GILLT_v16s1,
      GIR_BuildMI, /*InsnID*/3, /*Opcode*/TargetOpcode::COPY,
      GIR_AddTempRegister, /*InsnID*/3, /*TempRegID*/2, /*TempRegFlags*/RegState::Define,
      GIR_Copy, /*NewInsnID*/3, /*OldInsnID*/0, /*OpIdx*/2, // src2
      GIR_ConstrainSelectedInstOperands, /*InsnID*/3,
      GIR_BuildMI, /*InsnID*/2, /*Opcode*/TargetOpcode::COPY,
      GIR_AddTempRegister, /*InsnID*/2, /*TempRegID*/1, /*TempRegFlags*/RegState::Define,
      GIR_Copy, /*NewInsnID*/2, /*OldInsnID*/0, /*OpIdx*/1, // src1
      GIR_ConstrainSelectedInstOperands, /*InsnID*/2,
      GIR_BuildMI, /*InsnID*/1, /*Opcode*/X86::KORWrr,
      GIR_AddTempRegister, /*InsnID*/1, /*TempRegID*/0, /*TempRegFlags*/RegState::Define,
      GIR_AddTempRegister, /*InsnID*/1, /*TempRegID*/1, /*TempRegFlags*/0,
      GIR_AddTempRegister, /*InsnID*/1, /*TempRegID*/2, /*TempRegFlags*/0,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/1,
      GIR_BuildMI, /*InsnID*/0, /*Opcode*/TargetOpcode::COPY,
      GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
      GIR_AddTempRegister, /*InsnID*/0, /*TempRegID*/0, /*TempRegFlags*/0,
      GIR_EraseFromParent, /*InsnID*/0,
      GIR_ConstrainOperandRC, /*InsnID*/0, /*Op*/0, /*RC VK1*/8,
      // GIR_Coverage, 13741,
      GIR_Done,
    // Label 362: @9033
    GIM_Reject,
    // Label 310: @9034
    GIM_Try, /*On fail goto*//*Label 363*/ 9114,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v4s64,
      GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v4s64,
      GIM_Try, /*On fail goto*//*Label 364*/ 9067, // Rule ID 1551 //
        GIM_CheckFeatures, GIFBS_HasAVX2_NoVLX,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR256RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR256RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR256RegClassID,
        // (or:{ *:[v4i64] } VR256:{ *:[v4i64] }:$src1, VR256:{ *:[v4i64] }:$src2)  =>  (VPORYrr:{ *:[v4i64] } VR256:{ *:[v4i64] }:$src1, VR256:{ *:[v4i64] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VPORYrr,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 1551,
        GIR_Done,
      // Label 364: @9067
      GIM_Try, /*On fail goto*//*Label 365*/ 9090, // Rule ID 4925 //
        GIM_CheckFeatures, GIFBS_HasAVX512_HasVLX,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR256XRegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR256XRegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR256XRegClassID,
        // (or:{ *:[v4i64] } VR256X:{ *:[v4i64] }:$src1, VR256X:{ *:[v4i64] }:$src2)  =>  (VPORQZ256rr:{ *:[v4i64] } VR256X:{ *:[v4i64] }:$src1, VR256X:{ *:[v4i64] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VPORQZ256rr,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 4925,
        GIR_Done,
      // Label 365: @9090
      GIM_Try, /*On fail goto*//*Label 366*/ 9113, // Rule ID 12516 //
        GIM_CheckFeatures, GIFBS_HasAVX1Only,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR256RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR256RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR256RegClassID,
        // (or:{ *:[v4i64] } VR256:{ *:[v4i64] }:$src1, VR256:{ *:[v4i64] }:$src2)  =>  (VORPSYrr:{ *:[v4i64] } VR256:{ *:[v4i64] }:$src1, VR256:{ *:[v4i64] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VORPSYrr,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 12516,
        GIR_Done,
      // Label 366: @9113
      GIM_Reject,
    // Label 363: @9114
    GIM_Reject,
    // Label 311: @9115
    GIM_Try, /*On fail goto*//*Label 367*/ 9223,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v8s1,
      GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v8s1,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VK8RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VK8RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VK8RegClassID,
      GIM_Try, /*On fail goto*//*Label 368*/ 9148, // Rule ID 3605 //
        GIM_CheckFeatures, GIFBS_HasDQI,
        // (or:{ *:[v8i1] } VK8:{ *:[v8i1] }:$src1, VK8:{ *:[v8i1] }:$src2)  =>  (KORBrr:{ *:[v8i1] } VK8:{ *:[v8i1] }:$src1, VK8:{ *:[v8i1] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::KORBrr,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 3605,
        GIR_Done,
      // Label 368: @9148
      GIM_Try, /*On fail goto*//*Label 369*/ 9222, // Rule ID 13738 //
        GIM_CheckFeatures, GIFBS_NoDQI,
        // (or:{ *:[v8i1] } VK8:{ *:[v8i1] }:$src1, VK8:{ *:[v8i1] }:$src2)  =>  (COPY_TO_REGCLASS:{ *:[v8i1] } (KORWrr:{ *:[v16i1] } (COPY_TO_REGCLASS:{ *:[v16i1] } VK8:{ *:[v8i1] }:$src1, VK16:{ *:[i32] }), (COPY_TO_REGCLASS:{ *:[v16i1] } VK8:{ *:[v8i1] }:$src2, VK16:{ *:[i32] })), VK8:{ *:[i32] })
        GIR_MakeTempReg, /*TempRegID*/0, /*TypeID*/GILLT_v16s1,
        GIR_MakeTempReg, /*TempRegID*/1, /*TypeID*/GILLT_v16s1,
        GIR_MakeTempReg, /*TempRegID*/2, /*TypeID*/GILLT_v16s1,
        GIR_BuildMI, /*InsnID*/3, /*Opcode*/TargetOpcode::COPY,
        GIR_AddTempRegister, /*InsnID*/3, /*TempRegID*/2, /*TempRegFlags*/RegState::Define,
        GIR_Copy, /*NewInsnID*/3, /*OldInsnID*/0, /*OpIdx*/2, // src2
        GIR_ConstrainSelectedInstOperands, /*InsnID*/3,
        GIR_BuildMI, /*InsnID*/2, /*Opcode*/TargetOpcode::COPY,
        GIR_AddTempRegister, /*InsnID*/2, /*TempRegID*/1, /*TempRegFlags*/RegState::Define,
        GIR_Copy, /*NewInsnID*/2, /*OldInsnID*/0, /*OpIdx*/1, // src1
        GIR_ConstrainSelectedInstOperands, /*InsnID*/2,
        GIR_BuildMI, /*InsnID*/1, /*Opcode*/X86::KORWrr,
        GIR_AddTempRegister, /*InsnID*/1, /*TempRegID*/0, /*TempRegFlags*/RegState::Define,
        GIR_AddTempRegister, /*InsnID*/1, /*TempRegID*/1, /*TempRegFlags*/0,
        GIR_AddTempRegister, /*InsnID*/1, /*TempRegID*/2, /*TempRegFlags*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/1,
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/TargetOpcode::COPY,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_AddTempRegister, /*InsnID*/0, /*TempRegID*/0, /*TempRegFlags*/0,
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainOperandRC, /*InsnID*/0, /*Op*/0, /*RC VK8*/12,
        // GIR_Coverage, 13738,
        GIR_Done,
      // Label 369: @9222
      GIM_Reject,
    // Label 367: @9223
    GIM_Reject,
    // Label 312: @9224
    GIM_Try, /*On fail goto*//*Label 370*/ 9255, // Rule ID 4916 //
      GIM_CheckFeatures, GIFBS_HasAVX512,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v8s64,
      GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v8s64,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR512RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR512RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR512RegClassID,
      // (or:{ *:[v8i64] } VR512:{ *:[v8i64] }:$src1, VR512:{ *:[v8i64] }:$src2)  =>  (VPORQZrr:{ *:[v8i64] } VR512:{ *:[v8i64] }:$src1, VR512:{ *:[v8i64] }:$src2)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VPORQZrr,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 4916,
      GIR_Done,
    // Label 370: @9255
    GIM_Reject,
    // Label 313: @9256
    GIM_Try, /*On fail goto*//*Label 371*/ 9287, // Rule ID 3606 //
      GIM_CheckFeatures, GIFBS_HasAVX512,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v16s1,
      GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v16s1,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VK16RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VK16RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VK16RegClassID,
      // (or:{ *:[v16i1] } VK16:{ *:[v16i1] }:$src1, VK16:{ *:[v16i1] }:$src2)  =>  (KORWrr:{ *:[v16i1] } VK16:{ *:[v16i1] }:$src1, VK16:{ *:[v16i1] }:$src2)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::KORWrr,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 3606,
      GIR_Done,
    // Label 371: @9287
    GIM_Reject,
    // Label 314: @9288
    GIM_Try, /*On fail goto*//*Label 372*/ 9319, // Rule ID 3607 //
      GIM_CheckFeatures, GIFBS_HasBWI,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v32s1,
      GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v32s1,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VK32RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VK32RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VK32RegClassID,
      // (or:{ *:[v32i1] } VK32:{ *:[v32i1] }:$src1, VK32:{ *:[v32i1] }:$src2)  =>  (KORDrr:{ *:[v32i1] } VK32:{ *:[v32i1] }:$src1, VK32:{ *:[v32i1] }:$src2)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::KORDrr,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 3607,
      GIR_Done,
    // Label 372: @9319
    GIM_Reject,
    // Label 315: @9320
    GIM_Try, /*On fail goto*//*Label 373*/ 9351, // Rule ID 3608 //
      GIM_CheckFeatures, GIFBS_HasBWI,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v64s1,
      GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v64s1,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VK64RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VK64RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VK64RegClassID,
      // (or:{ *:[v64i1] } VK64:{ *:[v64i1] }:$src1, VK64:{ *:[v64i1] }:$src2)  =>  (KORQrr:{ *:[v64i1] } VK64:{ *:[v64i1] }:$src1, VK64:{ *:[v64i1] }:$src2)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::KORQrr,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 3608,
      GIR_Done,
    // Label 373: @9351
    GIM_Reject,
    // Label 316: @9352
    GIM_Reject,
    // Label 5: @9353
    GIM_SwitchType, /*MI*/0, /*Op*/0, /*[*/0, 24, /*)*//*default:*//*Label 388*/ 11325,
    /*GILLT_s1*//*Label 374*/ 9383,
    /*GILLT_s8*//*Label 375*/ 9782,
    /*GILLT_s16*//*Label 376*/ 9870,
    /*GILLT_s32*//*Label 377*/ 9991,
    /*GILLT_s64*//*Label 378*/ 10364, 0, 0,
    /*GILLT_v2s1*//*Label 379*/ 10740,
    /*GILLT_v2s64*//*Label 380*/ 10833,
    /*GILLT_v4s1*//*Label 381*/ 10914, 0,
    /*GILLT_v4s64*//*Label 382*/ 11007,
    /*GILLT_v8s1*//*Label 383*/ 11088, 0, 0,
    /*GILLT_v8s64*//*Label 384*/ 11197,
    /*GILLT_v16s1*//*Label 385*/ 11229, 0, 0, 0,
    /*GILLT_v32s1*//*Label 386*/ 11261, 0, 0,
    /*GILLT_v64s1*//*Label 387*/ 11293,
    // Label 374: @9383
    GIM_Try, /*On fail goto*//*Label 389*/ 9781,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_s1,
      GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_s1,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VK1RegClassID,
      GIM_Try, /*On fail goto*//*Label 390*/ 9498, // Rule ID 18297 //
        GIM_RecordInsn, /*DefineMI*/1, /*MI*/0, /*OpIdx*/1, // MIs[1]
        GIM_CheckOpcode, /*MI*/1, TargetOpcode::G_XOR,
        GIM_CheckType, /*MI*/1, /*Op*/1, /*Type*/GILLT_s1,
        GIM_CheckType, /*MI*/1, /*Op*/2, /*Type*/GILLT_s1,
        GIM_CheckRegBankForClass, /*MI*/1, /*Op*/1, /*RC*/X86::VK1RegClassID,
        GIM_CheckConstantInt, /*MI*/1, /*Op*/2, -1,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VK1RegClassID,
        GIM_CheckIsSafeToFold, /*InsnID*/1,
        // (xor:{ *:[v1i1] } (xor:{ *:[v1i1] } VK1:{ *:[v1i1] }:$src1, -1:{ *:[v1i1] }), VK1:{ *:[v1i1] }:$src2)  =>  (COPY_TO_REGCLASS:{ *:[v1i1] } (KXNORWrr:{ *:[v16i1] } (COPY_TO_REGCLASS:{ *:[v16i1] } VK1:{ *:[v1i1] }:$src1, VK16:{ *:[i32] }), (COPY_TO_REGCLASS:{ *:[v16i1] } VK1:{ *:[v1i1] }:$src2, VK16:{ *:[i32] })), VK1:{ *:[i32] })
        GIR_MakeTempReg, /*TempRegID*/0, /*TypeID*/GILLT_v16s1,
        GIR_MakeTempReg, /*TempRegID*/1, /*TypeID*/GILLT_v16s1,
        GIR_MakeTempReg, /*TempRegID*/2, /*TypeID*/GILLT_v16s1,
        GIR_BuildMI, /*InsnID*/3, /*Opcode*/TargetOpcode::COPY,
        GIR_AddTempRegister, /*InsnID*/3, /*TempRegID*/2, /*TempRegFlags*/RegState::Define,
        GIR_Copy, /*NewInsnID*/3, /*OldInsnID*/0, /*OpIdx*/2, // src2
        GIR_ConstrainSelectedInstOperands, /*InsnID*/3,
        GIR_BuildMI, /*InsnID*/2, /*Opcode*/TargetOpcode::COPY,
        GIR_AddTempRegister, /*InsnID*/2, /*TempRegID*/1, /*TempRegFlags*/RegState::Define,
        GIR_Copy, /*NewInsnID*/2, /*OldInsnID*/1, /*OpIdx*/1, // src1
        GIR_ConstrainSelectedInstOperands, /*InsnID*/2,
        GIR_BuildMI, /*InsnID*/1, /*Opcode*/X86::KXNORWrr,
        GIR_AddTempRegister, /*InsnID*/1, /*TempRegID*/0, /*TempRegFlags*/RegState::Define,
        GIR_AddTempRegister, /*InsnID*/1, /*TempRegID*/1, /*TempRegFlags*/0,
        GIR_AddTempRegister, /*InsnID*/1, /*TempRegID*/2, /*TempRegFlags*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/1,
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/TargetOpcode::COPY,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_AddTempRegister, /*InsnID*/0, /*TempRegID*/0, /*TempRegFlags*/0,
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainOperandRC, /*InsnID*/0, /*Op*/0, /*RC VK1*/8,
        // GIR_Coverage, 18297,
        GIR_Done,
      // Label 390: @9498
      GIM_Try, /*On fail goto*//*Label 391*/ 9599, // Rule ID 13743 //
        GIM_RecordInsn, /*DefineMI*/1, /*MI*/0, /*OpIdx*/1, // MIs[1]
        GIM_CheckOpcode, /*MI*/1, TargetOpcode::G_XOR,
        GIM_CheckType, /*MI*/1, /*Op*/1, /*Type*/GILLT_s1,
        GIM_CheckType, /*MI*/1, /*Op*/2, /*Type*/GILLT_s1,
        GIM_CheckRegBankForClass, /*MI*/1, /*Op*/1, /*RC*/X86::VK1RegClassID,
        GIM_CheckRegBankForClass, /*MI*/1, /*Op*/2, /*RC*/X86::VK1RegClassID,
        GIM_CheckConstantInt, /*MI*/0, /*Op*/2, -1,
        GIM_CheckIsSafeToFold, /*InsnID*/1,
        // (xor:{ *:[v1i1] } (xor:{ *:[v1i1] } VK1:{ *:[v1i1] }:$src1, VK1:{ *:[v1i1] }:$src2), -1:{ *:[v1i1] })  =>  (COPY_TO_REGCLASS:{ *:[v1i1] } (KXNORWrr:{ *:[v16i1] } (COPY_TO_REGCLASS:{ *:[v16i1] } VK1:{ *:[v1i1] }:$src1, VK16:{ *:[i32] }), (COPY_TO_REGCLASS:{ *:[v16i1] } VK1:{ *:[v1i1] }:$src2, VK16:{ *:[i32] })), VK1:{ *:[i32] })
        GIR_MakeTempReg, /*TempRegID*/0, /*TypeID*/GILLT_v16s1,
        GIR_MakeTempReg, /*TempRegID*/1, /*TypeID*/GILLT_v16s1,
        GIR_MakeTempReg, /*TempRegID*/2, /*TypeID*/GILLT_v16s1,
        GIR_BuildMI, /*InsnID*/3, /*Opcode*/TargetOpcode::COPY,
        GIR_AddTempRegister, /*InsnID*/3, /*TempRegID*/2, /*TempRegFlags*/RegState::Define,
        GIR_Copy, /*NewInsnID*/3, /*OldInsnID*/1, /*OpIdx*/2, // src2
        GIR_ConstrainSelectedInstOperands, /*InsnID*/3,
        GIR_BuildMI, /*InsnID*/2, /*Opcode*/TargetOpcode::COPY,
        GIR_AddTempRegister, /*InsnID*/2, /*TempRegID*/1, /*TempRegFlags*/RegState::Define,
        GIR_Copy, /*NewInsnID*/2, /*OldInsnID*/1, /*OpIdx*/1, // src1
        GIR_ConstrainSelectedInstOperands, /*InsnID*/2,
        GIR_BuildMI, /*InsnID*/1, /*Opcode*/X86::KXNORWrr,
        GIR_AddTempRegister, /*InsnID*/1, /*TempRegID*/0, /*TempRegFlags*/RegState::Define,
        GIR_AddTempRegister, /*InsnID*/1, /*TempRegID*/1, /*TempRegFlags*/0,
        GIR_AddTempRegister, /*InsnID*/1, /*TempRegID*/2, /*TempRegFlags*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/1,
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/TargetOpcode::COPY,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_AddTempRegister, /*InsnID*/0, /*TempRegID*/0, /*TempRegFlags*/0,
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainOperandRC, /*InsnID*/0, /*Op*/0, /*RC VK1*/8,
        // GIR_Coverage, 13743,
        GIR_Done,
      // Label 391: @9599
      GIM_Try, /*On fail goto*//*Label 392*/ 9700, // Rule ID 18298 //
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VK1RegClassID,
        GIM_RecordInsn, /*DefineMI*/1, /*MI*/0, /*OpIdx*/2, // MIs[1]
        GIM_CheckOpcode, /*MI*/1, TargetOpcode::G_XOR,
        GIM_CheckType, /*MI*/1, /*Op*/1, /*Type*/GILLT_s1,
        GIM_CheckType, /*MI*/1, /*Op*/2, /*Type*/GILLT_s1,
        GIM_CheckRegBankForClass, /*MI*/1, /*Op*/1, /*RC*/X86::VK1RegClassID,
        GIM_CheckConstantInt, /*MI*/1, /*Op*/2, -1,
        GIM_CheckIsSafeToFold, /*InsnID*/1,
        // (xor:{ *:[v1i1] } VK1:{ *:[v1i1] }:$src2, (xor:{ *:[v1i1] } VK1:{ *:[v1i1] }:$src1, -1:{ *:[v1i1] }))  =>  (COPY_TO_REGCLASS:{ *:[v1i1] } (KXNORWrr:{ *:[v16i1] } (COPY_TO_REGCLASS:{ *:[v16i1] } VK1:{ *:[v1i1] }:$src1, VK16:{ *:[i32] }), (COPY_TO_REGCLASS:{ *:[v16i1] } VK1:{ *:[v1i1] }:$src2, VK16:{ *:[i32] })), VK1:{ *:[i32] })
        GIR_MakeTempReg, /*TempRegID*/0, /*TypeID*/GILLT_v16s1,
        GIR_MakeTempReg, /*TempRegID*/1, /*TypeID*/GILLT_v16s1,
        GIR_MakeTempReg, /*TempRegID*/2, /*TypeID*/GILLT_v16s1,
        GIR_BuildMI, /*InsnID*/3, /*Opcode*/TargetOpcode::COPY,
        GIR_AddTempRegister, /*InsnID*/3, /*TempRegID*/2, /*TempRegFlags*/RegState::Define,
        GIR_Copy, /*NewInsnID*/3, /*OldInsnID*/0, /*OpIdx*/1, // src2
        GIR_ConstrainSelectedInstOperands, /*InsnID*/3,
        GIR_BuildMI, /*InsnID*/2, /*Opcode*/TargetOpcode::COPY,
        GIR_AddTempRegister, /*InsnID*/2, /*TempRegID*/1, /*TempRegFlags*/RegState::Define,
        GIR_Copy, /*NewInsnID*/2, /*OldInsnID*/1, /*OpIdx*/1, // src1
        GIR_ConstrainSelectedInstOperands, /*InsnID*/2,
        GIR_BuildMI, /*InsnID*/1, /*Opcode*/X86::KXNORWrr,
        GIR_AddTempRegister, /*InsnID*/1, /*TempRegID*/0, /*TempRegFlags*/RegState::Define,
        GIR_AddTempRegister, /*InsnID*/1, /*TempRegID*/1, /*TempRegFlags*/0,
        GIR_AddTempRegister, /*InsnID*/1, /*TempRegID*/2, /*TempRegFlags*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/1,
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/TargetOpcode::COPY,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_AddTempRegister, /*InsnID*/0, /*TempRegID*/0, /*TempRegFlags*/0,
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainOperandRC, /*InsnID*/0, /*Op*/0, /*RC VK1*/8,
        // GIR_Coverage, 18298,
        GIR_Done,
      // Label 392: @9700
      GIM_Try, /*On fail goto*//*Label 393*/ 9780, // Rule ID 13747 //
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VK1RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VK1RegClassID,
        // (xor:{ *:[v1i1] } VK1:{ *:[v1i1] }:$src1, VK1:{ *:[v1i1] }:$src2)  =>  (COPY_TO_REGCLASS:{ *:[v1i1] } (KXORWrr:{ *:[v16i1] } (COPY_TO_REGCLASS:{ *:[v16i1] } VK1:{ *:[v1i1] }:$src1, VK16:{ *:[i32] }), (COPY_TO_REGCLASS:{ *:[v16i1] } VK1:{ *:[v1i1] }:$src2, VK16:{ *:[i32] })), VK1:{ *:[i32] })
        GIR_MakeTempReg, /*TempRegID*/0, /*TypeID*/GILLT_v16s1,
        GIR_MakeTempReg, /*TempRegID*/1, /*TypeID*/GILLT_v16s1,
        GIR_MakeTempReg, /*TempRegID*/2, /*TypeID*/GILLT_v16s1,
        GIR_BuildMI, /*InsnID*/3, /*Opcode*/TargetOpcode::COPY,
        GIR_AddTempRegister, /*InsnID*/3, /*TempRegID*/2, /*TempRegFlags*/RegState::Define,
        GIR_Copy, /*NewInsnID*/3, /*OldInsnID*/0, /*OpIdx*/2, // src2
        GIR_ConstrainSelectedInstOperands, /*InsnID*/3,
        GIR_BuildMI, /*InsnID*/2, /*Opcode*/TargetOpcode::COPY,
        GIR_AddTempRegister, /*InsnID*/2, /*TempRegID*/1, /*TempRegFlags*/RegState::Define,
        GIR_Copy, /*NewInsnID*/2, /*OldInsnID*/0, /*OpIdx*/1, // src1
        GIR_ConstrainSelectedInstOperands, /*InsnID*/2,
        GIR_BuildMI, /*InsnID*/1, /*Opcode*/X86::KXORWrr,
        GIR_AddTempRegister, /*InsnID*/1, /*TempRegID*/0, /*TempRegFlags*/RegState::Define,
        GIR_AddTempRegister, /*InsnID*/1, /*TempRegID*/1, /*TempRegFlags*/0,
        GIR_AddTempRegister, /*InsnID*/1, /*TempRegID*/2, /*TempRegFlags*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/1,
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/TargetOpcode::COPY,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_AddTempRegister, /*InsnID*/0, /*TempRegID*/0, /*TempRegFlags*/0,
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainOperandRC, /*InsnID*/0, /*Op*/0, /*RC VK1*/8,
        // GIR_Coverage, 13747,
        GIR_Done,
      // Label 393: @9780
      GIM_Reject,
    // Label 389: @9781
    GIM_Reject,
    // Label 375: @9782
    GIM_Try, /*On fail goto*//*Label 394*/ 9869,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_s8,
      GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_s8,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::GR8RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::GR8RegClassID,
      GIM_Try, /*On fail goto*//*Label 395*/ 9822, // Rule ID 152 //
        GIM_CheckConstantInt, /*MI*/0, /*Op*/2, -1,
        // (xor:{ *:[i8] } GR8:{ *:[i8] }:$src1, -1:{ *:[i8] })  =>  (NOT8r:{ *:[i8] } GR8:{ *:[i8] }:$src1)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::NOT8r,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // src1
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 152,
        GIR_Done,
      // Label 395: @9822
      GIM_Try, /*On fail goto*//*Label 396*/ 9852, // Rule ID 16179 //
        GIM_RecordInsn, /*DefineMI*/1, /*MI*/0, /*OpIdx*/2, // MIs[1]
        GIM_CheckOpcode, /*MI*/1, TargetOpcode::G_CONSTANT,
        // MIs[1] Operand 1
        // No operand predicates
        GIM_CheckIsSafeToFold, /*InsnID*/1,
        // (xor:{ *:[i8] } GR8:{ *:[i8] }:$src1, (imm:{ *:[i8] }):$src2)  =>  (XOR8ri:{ *:[i8] }:{ *:[i32] } GR8:{ *:[i8] }:$src1, (imm:{ *:[i8] }):$src2)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::XOR8ri,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // src1
        GIR_CopyConstantAsSImm, /*NewInsnID*/0, /*OldInsnID*/1, // src2
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 16179,
        GIR_Done,
      // Label 396: @9852
      GIM_Try, /*On fail goto*//*Label 397*/ 9868, // Rule ID 16171 //
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::GR8RegClassID,
        // (xor:{ *:[i8] } GR8:{ *:[i8] }:$src1, GR8:{ *:[i8] }:$src2)  =>  (XOR8rr:{ *:[i8] }:{ *:[i32] } GR8:{ *:[i8] }:$src1, GR8:{ *:[i8] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::XOR8rr,
        GIR_AddImplicitDef, /*InsnID*/0, X86::EFLAGS,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 16171,
        GIR_Done,
      // Label 397: @9868
      GIM_Reject,
    // Label 394: @9869
    GIM_Reject,
    // Label 376: @9870
    GIM_Try, /*On fail goto*//*Label 398*/ 9990,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_s16,
      GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_s16,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::GR16RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::GR16RegClassID,
      GIM_Try, /*On fail goto*//*Label 399*/ 9910, // Rule ID 153 //
        GIM_CheckConstantInt, /*MI*/0, /*Op*/2, -1,
        // (xor:{ *:[i16] } GR16:{ *:[i16] }:$src1, -1:{ *:[i16] })  =>  (NOT16r:{ *:[i16] } GR16:{ *:[i16] }:$src1)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::NOT16r,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // src1
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 153,
        GIR_Done,
      // Label 399: @9910
      GIM_Try, /*On fail goto*//*Label 400*/ 9943, // Rule ID 16182 //
        GIM_RecordInsn, /*DefineMI*/1, /*MI*/0, /*OpIdx*/2, // MIs[1]
        GIM_CheckOpcode, /*MI*/1, TargetOpcode::G_CONSTANT,
        GIM_CheckI64ImmPredicate, /*MI*/1, /*Predicate*/GIPFP_I64_Predicate_i16immSExt8,
        // MIs[1] Operand 1
        // No operand predicates
        GIM_CheckIsSafeToFold, /*InsnID*/1,
        // (xor:{ *:[i16] } GR16:{ *:[i16] }:$src1, (imm:{ *:[i16] })<<P:Predicate_i16immSExt8>>:$src2)  =>  (XOR16ri8:{ *:[i16] }:{ *:[i32] } GR16:{ *:[i16] }:$src1, (imm:{ *:[i16] })<<P:Predicate_i16immSExt8>>:$src2)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::XOR16ri8,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // src1
        GIR_CopyConstantAsSImm, /*NewInsnID*/0, /*OldInsnID*/1, // src2
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 16182,
        GIR_Done,
      // Label 400: @9943
      GIM_Try, /*On fail goto*//*Label 401*/ 9973, // Rule ID 16180 //
        GIM_RecordInsn, /*DefineMI*/1, /*MI*/0, /*OpIdx*/2, // MIs[1]
        GIM_CheckOpcode, /*MI*/1, TargetOpcode::G_CONSTANT,
        // MIs[1] Operand 1
        // No operand predicates
        GIM_CheckIsSafeToFold, /*InsnID*/1,
        // (xor:{ *:[i16] } GR16:{ *:[i16] }:$src1, (imm:{ *:[i16] }):$src2)  =>  (XOR16ri:{ *:[i16] }:{ *:[i32] } GR16:{ *:[i16] }:$src1, (imm:{ *:[i16] }):$src2)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::XOR16ri,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // src1
        GIR_CopyConstantAsSImm, /*NewInsnID*/0, /*OldInsnID*/1, // src2
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 16180,
        GIR_Done,
      // Label 401: @9973
      GIM_Try, /*On fail goto*//*Label 402*/ 9989, // Rule ID 16172 //
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::GR16RegClassID,
        // (xor:{ *:[i16] } GR16:{ *:[i16] }:$src1, GR16:{ *:[i16] }:$src2)  =>  (XOR16rr:{ *:[i16] }:{ *:[i32] } GR16:{ *:[i16] }:$src1, GR16:{ *:[i16] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::XOR16rr,
        GIR_AddImplicitDef, /*InsnID*/0, X86::EFLAGS,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 16172,
        GIR_Done,
      // Label 402: @9989
      GIM_Reject,
    // Label 398: @9990
    GIM_Reject,
    // Label 377: @9991
    GIM_Try, /*On fail goto*//*Label 403*/ 10363,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_s32,
      GIM_Try, /*On fail goto*//*Label 404*/ 10051, // Rule ID 18156 //
        GIM_CheckFeatures, GIFBS_HasTBM,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::GR32RegClassID,
        GIM_RecordInsn, /*DefineMI*/1, /*MI*/0, /*OpIdx*/1, // MIs[1]
        GIM_CheckOpcode, /*MI*/1, TargetOpcode::G_ADD,
        GIM_CheckType, /*MI*/1, /*Op*/1, /*Type*/GILLT_s32,
        GIM_CheckType, /*MI*/1, /*Op*/2, /*Type*/GILLT_s32,
        GIM_CheckRegBankForClass, /*MI*/1, /*Op*/1, /*RC*/X86::GR32RegClassID,
        GIM_CheckConstantInt, /*MI*/1, /*Op*/2, 1,
        // MIs[0] src
        GIM_CheckIsSameOperand, /*MI*/0, /*OpIdx*/2, /*OtherMI*/1, /*OtherOpIdx*/1,
        GIM_CheckIsSafeToFold, /*InsnID*/1,
        // (xor:{ *:[i32] } (add:{ *:[i32] } GR32:{ *:[i32] }:$src, 1:{ *:[i32] }), GR32:{ *:[i32] }:$src)  =>  (BLCMSK32rr:{ *:[i32] }:{ *:[i32] } GR32:{ *:[i32] }:$src)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::BLCMSK32rr,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/1, /*OpIdx*/1, // src
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 18156,
        GIR_Done,
      // Label 404: @10051
      GIM_Try, /*On fail goto*//*Label 405*/ 10105, // Rule ID 18676 //
        GIM_CheckFeatures, GIFBS_HasBMI,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::GR32RegClassID,
        GIM_RecordInsn, /*DefineMI*/1, /*MI*/0, /*OpIdx*/1, // MIs[1]
        GIM_CheckOpcode, /*MI*/1, TargetOpcode::G_ADD,
        GIM_CheckType, /*MI*/1, /*Op*/1, /*Type*/GILLT_s32,
        GIM_CheckType, /*MI*/1, /*Op*/2, /*Type*/GILLT_s32,
        GIM_CheckRegBankForClass, /*MI*/1, /*Op*/1, /*RC*/X86::GR32RegClassID,
        GIM_CheckConstantInt, /*MI*/1, /*Op*/2, -1,
        // MIs[0] src
        GIM_CheckIsSameOperand, /*MI*/0, /*OpIdx*/2, /*OtherMI*/1, /*OtherOpIdx*/1,
        GIM_CheckIsSafeToFold, /*InsnID*/1,
        // (xor:{ *:[i32] } (add:{ *:[i32] } GR32:{ *:[i32] }:$src, -1:{ *:[i32] }), GR32:{ *:[i32] }:$src)  =>  (BLSMSK32rr:{ *:[i32] }:{ *:[i32] } GR32:{ *:[i32] }:$src)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::BLSMSK32rr,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/1, /*OpIdx*/1, // src
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 18676,
        GIR_Done,
      // Label 405: @10105
      GIM_Try, /*On fail goto*//*Label 406*/ 10159, // Rule ID 12229 //
        GIM_CheckFeatures, GIFBS_HasTBM,
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_s32,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::GR32RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::GR32RegClassID,
        GIM_RecordInsn, /*DefineMI*/1, /*MI*/0, /*OpIdx*/2, // MIs[1]
        GIM_CheckOpcode, /*MI*/1, TargetOpcode::G_ADD,
        GIM_CheckType, /*MI*/1, /*Op*/2, /*Type*/GILLT_s32,
        // MIs[1] src
        GIM_CheckIsSameOperand, /*MI*/1, /*OpIdx*/1, /*OtherMI*/0, /*OtherOpIdx*/1,
        GIM_CheckConstantInt, /*MI*/1, /*Op*/2, 1,
        GIM_CheckIsSafeToFold, /*InsnID*/1,
        // (xor:{ *:[i32] } GR32:{ *:[i32] }:$src, (add:{ *:[i32] } GR32:{ *:[i32] }:$src, 1:{ *:[i32] }))  =>  (BLCMSK32rr:{ *:[i32] }:{ *:[i32] } GR32:{ *:[i32] }:$src)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::BLCMSK32rr,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // src
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 12229,
        GIR_Done,
      // Label 406: @10159
      GIM_Try, /*On fail goto*//*Label 407*/ 10213, // Rule ID 16395 //
        GIM_CheckFeatures, GIFBS_HasBMI,
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_s32,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::GR32RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::GR32RegClassID,
        GIM_RecordInsn, /*DefineMI*/1, /*MI*/0, /*OpIdx*/2, // MIs[1]
        GIM_CheckOpcode, /*MI*/1, TargetOpcode::G_ADD,
        GIM_CheckType, /*MI*/1, /*Op*/2, /*Type*/GILLT_s32,
        // MIs[1] src
        GIM_CheckIsSameOperand, /*MI*/1, /*OpIdx*/1, /*OtherMI*/0, /*OtherOpIdx*/1,
        GIM_CheckConstantInt, /*MI*/1, /*Op*/2, -1,
        GIM_CheckIsSafeToFold, /*InsnID*/1,
        // (xor:{ *:[i32] } GR32:{ *:[i32] }:$src, (add:{ *:[i32] } GR32:{ *:[i32] }:$src, -1:{ *:[i32] }))  =>  (BLSMSK32rr:{ *:[i32] }:{ *:[i32] } GR32:{ *:[i32] }:$src)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::BLSMSK32rr,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // src
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 16395,
        GIR_Done,
      // Label 407: @10213
      GIM_Try, /*On fail goto*//*Label 408*/ 10247, // Rule ID 154 //
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_s32,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::GR32RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::GR32RegClassID,
        GIM_CheckConstantInt, /*MI*/0, /*Op*/2, -1,
        // (xor:{ *:[i32] } GR32:{ *:[i32] }:$src1, -1:{ *:[i32] })  =>  (NOT32r:{ *:[i32] } GR32:{ *:[i32] }:$src1)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::NOT32r,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // src1
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 154,
        GIR_Done,
      // Label 408: @10247
      GIM_Try, /*On fail goto*//*Label 409*/ 10292, // Rule ID 16183 //
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_s32,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::GR32RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::GR32RegClassID,
        GIM_RecordInsn, /*DefineMI*/1, /*MI*/0, /*OpIdx*/2, // MIs[1]
        GIM_CheckOpcode, /*MI*/1, TargetOpcode::G_CONSTANT,
        GIM_CheckI64ImmPredicate, /*MI*/1, /*Predicate*/GIPFP_I64_Predicate_i32immSExt8,
        // MIs[1] Operand 1
        // No operand predicates
        GIM_CheckIsSafeToFold, /*InsnID*/1,
        // (xor:{ *:[i32] } GR32:{ *:[i32] }:$src1, (imm:{ *:[i32] })<<P:Predicate_i32immSExt8>>:$src2)  =>  (XOR32ri8:{ *:[i32] }:{ *:[i32] } GR32:{ *:[i32] }:$src1, (imm:{ *:[i32] })<<P:Predicate_i32immSExt8>>:$src2)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::XOR32ri8,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // src1
        GIR_CopyConstantAsSImm, /*NewInsnID*/0, /*OldInsnID*/1, // src2
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 16183,
        GIR_Done,
      // Label 409: @10292
      GIM_Try, /*On fail goto*//*Label 410*/ 10334, // Rule ID 16181 //
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_s32,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::GR32RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::GR32RegClassID,
        GIM_RecordInsn, /*DefineMI*/1, /*MI*/0, /*OpIdx*/2, // MIs[1]
        GIM_CheckOpcode, /*MI*/1, TargetOpcode::G_CONSTANT,
        // MIs[1] Operand 1
        // No operand predicates
        GIM_CheckIsSafeToFold, /*InsnID*/1,
        // (xor:{ *:[i32] } GR32:{ *:[i32] }:$src1, (imm:{ *:[i32] }):$src2)  =>  (XOR32ri:{ *:[i32] }:{ *:[i32] } GR32:{ *:[i32] }:$src1, (imm:{ *:[i32] }):$src2)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::XOR32ri,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // src1
        GIR_CopyConstantAsSImm, /*NewInsnID*/0, /*OldInsnID*/1, // src2
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 16181,
        GIR_Done,
      // Label 410: @10334
      GIM_Try, /*On fail goto*//*Label 411*/ 10362, // Rule ID 16173 //
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_s32,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::GR32RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::GR32RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::GR32RegClassID,
        // (xor:{ *:[i32] } GR32:{ *:[i32] }:$src1, GR32:{ *:[i32] }:$src2)  =>  (XOR32rr:{ *:[i32] }:{ *:[i32] } GR32:{ *:[i32] }:$src1, GR32:{ *:[i32] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::XOR32rr,
        GIR_AddImplicitDef, /*InsnID*/0, X86::EFLAGS,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 16173,
        GIR_Done,
      // Label 411: @10362
      GIM_Reject,
    // Label 403: @10363
    GIM_Reject,
    // Label 378: @10364
    GIM_Try, /*On fail goto*//*Label 412*/ 10739,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_s64,
      GIM_Try, /*On fail goto*//*Label 413*/ 10424, // Rule ID 18157 //
        GIM_CheckFeatures, GIFBS_HasTBM,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::GR64RegClassID,
        GIM_RecordInsn, /*DefineMI*/1, /*MI*/0, /*OpIdx*/1, // MIs[1]
        GIM_CheckOpcode, /*MI*/1, TargetOpcode::G_ADD,
        GIM_CheckType, /*MI*/1, /*Op*/1, /*Type*/GILLT_s64,
        GIM_CheckType, /*MI*/1, /*Op*/2, /*Type*/GILLT_s64,
        GIM_CheckRegBankForClass, /*MI*/1, /*Op*/1, /*RC*/X86::GR64RegClassID,
        GIM_CheckConstantInt, /*MI*/1, /*Op*/2, 1,
        // MIs[0] src
        GIM_CheckIsSameOperand, /*MI*/0, /*OpIdx*/2, /*OtherMI*/1, /*OtherOpIdx*/1,
        GIM_CheckIsSafeToFold, /*InsnID*/1,
        // (xor:{ *:[i64] } (add:{ *:[i64] } GR64:{ *:[i64] }:$src, 1:{ *:[i64] }), GR64:{ *:[i64] }:$src)  =>  (BLCMSK64rr:{ *:[i64] }:{ *:[i32] } GR64:{ *:[i64] }:$src)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::BLCMSK64rr,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/1, /*OpIdx*/1, // src
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 18157,
        GIR_Done,
      // Label 413: @10424
      GIM_Try, /*On fail goto*//*Label 414*/ 10478, // Rule ID 18677 //
        GIM_CheckFeatures, GIFBS_HasBMI,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::GR64RegClassID,
        GIM_RecordInsn, /*DefineMI*/1, /*MI*/0, /*OpIdx*/1, // MIs[1]
        GIM_CheckOpcode, /*MI*/1, TargetOpcode::G_ADD,
        GIM_CheckType, /*MI*/1, /*Op*/1, /*Type*/GILLT_s64,
        GIM_CheckType, /*MI*/1, /*Op*/2, /*Type*/GILLT_s64,
        GIM_CheckRegBankForClass, /*MI*/1, /*Op*/1, /*RC*/X86::GR64RegClassID,
        GIM_CheckConstantInt, /*MI*/1, /*Op*/2, -1,
        // MIs[0] src
        GIM_CheckIsSameOperand, /*MI*/0, /*OpIdx*/2, /*OtherMI*/1, /*OtherOpIdx*/1,
        GIM_CheckIsSafeToFold, /*InsnID*/1,
        // (xor:{ *:[i64] } (add:{ *:[i64] } GR64:{ *:[i64] }:$src, -1:{ *:[i64] }), GR64:{ *:[i64] }:$src)  =>  (BLSMSK64rr:{ *:[i64] }:{ *:[i32] } GR64:{ *:[i64] }:$src)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::BLSMSK64rr,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/1, /*OpIdx*/1, // src
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 18677,
        GIR_Done,
      // Label 414: @10478
      GIM_Try, /*On fail goto*//*Label 415*/ 10532, // Rule ID 12230 //
        GIM_CheckFeatures, GIFBS_HasTBM,
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_s64,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::GR64RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::GR64RegClassID,
        GIM_RecordInsn, /*DefineMI*/1, /*MI*/0, /*OpIdx*/2, // MIs[1]
        GIM_CheckOpcode, /*MI*/1, TargetOpcode::G_ADD,
        GIM_CheckType, /*MI*/1, /*Op*/2, /*Type*/GILLT_s64,
        // MIs[1] src
        GIM_CheckIsSameOperand, /*MI*/1, /*OpIdx*/1, /*OtherMI*/0, /*OtherOpIdx*/1,
        GIM_CheckConstantInt, /*MI*/1, /*Op*/2, 1,
        GIM_CheckIsSafeToFold, /*InsnID*/1,
        // (xor:{ *:[i64] } GR64:{ *:[i64] }:$src, (add:{ *:[i64] } GR64:{ *:[i64] }:$src, 1:{ *:[i64] }))  =>  (BLCMSK64rr:{ *:[i64] }:{ *:[i32] } GR64:{ *:[i64] }:$src)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::BLCMSK64rr,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // src
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 12230,
        GIR_Done,
      // Label 415: @10532
      GIM_Try, /*On fail goto*//*Label 416*/ 10586, // Rule ID 16396 //
        GIM_CheckFeatures, GIFBS_HasBMI,
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_s64,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::GR64RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::GR64RegClassID,
        GIM_RecordInsn, /*DefineMI*/1, /*MI*/0, /*OpIdx*/2, // MIs[1]
        GIM_CheckOpcode, /*MI*/1, TargetOpcode::G_ADD,
        GIM_CheckType, /*MI*/1, /*Op*/2, /*Type*/GILLT_s64,
        // MIs[1] src
        GIM_CheckIsSameOperand, /*MI*/1, /*OpIdx*/1, /*OtherMI*/0, /*OtherOpIdx*/1,
        GIM_CheckConstantInt, /*MI*/1, /*Op*/2, -1,
        GIM_CheckIsSafeToFold, /*InsnID*/1,
        // (xor:{ *:[i64] } GR64:{ *:[i64] }:$src, (add:{ *:[i64] } GR64:{ *:[i64] }:$src, -1:{ *:[i64] }))  =>  (BLSMSK64rr:{ *:[i64] }:{ *:[i32] } GR64:{ *:[i64] }:$src)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::BLSMSK64rr,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // src
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 16396,
        GIR_Done,
      // Label 416: @10586
      GIM_Try, /*On fail goto*//*Label 417*/ 10620, // Rule ID 155 //
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_s64,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::GR64RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::GR64RegClassID,
        GIM_CheckConstantInt, /*MI*/0, /*Op*/2, -1,
        // (xor:{ *:[i64] } GR64:{ *:[i64] }:$src1, -1:{ *:[i64] })  =>  (NOT64r:{ *:[i64] } GR64:{ *:[i64] }:$src1)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::NOT64r,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // src1
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 155,
        GIR_Done,
      // Label 417: @10620
      GIM_Try, /*On fail goto*//*Label 418*/ 10665, // Rule ID 16184 //
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_s64,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::GR64RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::GR64RegClassID,
        GIM_RecordInsn, /*DefineMI*/1, /*MI*/0, /*OpIdx*/2, // MIs[1]
        GIM_CheckOpcode, /*MI*/1, TargetOpcode::G_CONSTANT,
        GIM_CheckI64ImmPredicate, /*MI*/1, /*Predicate*/GIPFP_I64_Predicate_i64immSExt8,
        // MIs[1] Operand 1
        // No operand predicates
        GIM_CheckIsSafeToFold, /*InsnID*/1,
        // (xor:{ *:[i64] } GR64:{ *:[i64] }:$src1, (imm:{ *:[i64] })<<P:Predicate_i64immSExt8>>:$src2)  =>  (XOR64ri8:{ *:[i64] }:{ *:[i32] } GR64:{ *:[i64] }:$src1, (imm:{ *:[i64] })<<P:Predicate_i64immSExt8>>:$src2)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::XOR64ri8,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // src1
        GIR_CopyConstantAsSImm, /*NewInsnID*/0, /*OldInsnID*/1, // src2
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 16184,
        GIR_Done,
      // Label 418: @10665
      GIM_Try, /*On fail goto*//*Label 419*/ 10710, // Rule ID 16185 //
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_s64,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::GR64RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::GR64RegClassID,
        GIM_RecordInsn, /*DefineMI*/1, /*MI*/0, /*OpIdx*/2, // MIs[1]
        GIM_CheckOpcode, /*MI*/1, TargetOpcode::G_CONSTANT,
        GIM_CheckI64ImmPredicate, /*MI*/1, /*Predicate*/GIPFP_I64_Predicate_i64immSExt32,
        // MIs[1] Operand 1
        // No operand predicates
        GIM_CheckIsSafeToFold, /*InsnID*/1,
        // (xor:{ *:[i64] } GR64:{ *:[i64] }:$src1, (imm:{ *:[i64] })<<P:Predicate_i64immSExt32>>:$src2)  =>  (XOR64ri32:{ *:[i64] }:{ *:[i32] } GR64:{ *:[i64] }:$src1, (imm:{ *:[i64] })<<P:Predicate_i64immSExt32>>:$src2)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::XOR64ri32,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // src1
        GIR_CopyConstantAsSImm, /*NewInsnID*/0, /*OldInsnID*/1, // src2
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 16185,
        GIR_Done,
      // Label 419: @10710
      GIM_Try, /*On fail goto*//*Label 420*/ 10738, // Rule ID 16174 //
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_s64,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::GR64RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::GR64RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::GR64RegClassID,
        // (xor:{ *:[i64] } GR64:{ *:[i64] }:$src1, GR64:{ *:[i64] }:$src2)  =>  (XOR64rr:{ *:[i64] }:{ *:[i32] } GR64:{ *:[i64] }:$src1, GR64:{ *:[i64] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::XOR64rr,
        GIR_AddImplicitDef, /*InsnID*/0, X86::EFLAGS,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 16174,
        GIR_Done,
      // Label 420: @10738
      GIM_Reject,
    // Label 412: @10739
    GIM_Reject,
    // Label 379: @10740
    GIM_Try, /*On fail goto*//*Label 421*/ 10832, // Rule ID 13748 //
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v2s1,
      GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v2s1,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VK1RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VK2RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VK2RegClassID,
      // (xor:{ *:[v2i1] } VK2:{ *:[v2i1] }:$src1, VK2:{ *:[v2i1] }:$src2)  =>  (COPY_TO_REGCLASS:{ *:[v2i1] } (KXORWrr:{ *:[v16i1] } (COPY_TO_REGCLASS:{ *:[v16i1] } VK2:{ *:[v2i1] }:$src1, VK16:{ *:[i32] }), (COPY_TO_REGCLASS:{ *:[v16i1] } VK2:{ *:[v2i1] }:$src2, VK16:{ *:[i32] })), VK1:{ *:[i32] })
      GIR_MakeTempReg, /*TempRegID*/0, /*TypeID*/GILLT_v16s1,
      GIR_MakeTempReg, /*TempRegID*/1, /*TypeID*/GILLT_v16s1,
      GIR_MakeTempReg, /*TempRegID*/2, /*TypeID*/GILLT_v16s1,
      GIR_BuildMI, /*InsnID*/3, /*Opcode*/TargetOpcode::COPY,
      GIR_AddTempRegister, /*InsnID*/3, /*TempRegID*/2, /*TempRegFlags*/RegState::Define,
      GIR_Copy, /*NewInsnID*/3, /*OldInsnID*/0, /*OpIdx*/2, // src2
      GIR_ConstrainSelectedInstOperands, /*InsnID*/3,
      GIR_BuildMI, /*InsnID*/2, /*Opcode*/TargetOpcode::COPY,
      GIR_AddTempRegister, /*InsnID*/2, /*TempRegID*/1, /*TempRegFlags*/RegState::Define,
      GIR_Copy, /*NewInsnID*/2, /*OldInsnID*/0, /*OpIdx*/1, // src1
      GIR_ConstrainSelectedInstOperands, /*InsnID*/2,
      GIR_BuildMI, /*InsnID*/1, /*Opcode*/X86::KXORWrr,
      GIR_AddTempRegister, /*InsnID*/1, /*TempRegID*/0, /*TempRegFlags*/RegState::Define,
      GIR_AddTempRegister, /*InsnID*/1, /*TempRegID*/1, /*TempRegFlags*/0,
      GIR_AddTempRegister, /*InsnID*/1, /*TempRegID*/2, /*TempRegFlags*/0,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/1,
      GIR_BuildMI, /*InsnID*/0, /*Opcode*/TargetOpcode::COPY,
      GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
      GIR_AddTempRegister, /*InsnID*/0, /*TempRegID*/0, /*TempRegFlags*/0,
      GIR_EraseFromParent, /*InsnID*/0,
      GIR_ConstrainOperandRC, /*InsnID*/0, /*Op*/0, /*RC VK1*/8,
      // GIR_Coverage, 13748,
      GIR_Done,
    // Label 421: @10832
    GIM_Reject,
    // Label 380: @10833
    GIM_Try, /*On fail goto*//*Label 422*/ 10913,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v2s64,
      GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v2s64,
      GIM_Try, /*On fail goto*//*Label 423*/ 10866, // Rule ID 1553 //
        GIM_CheckFeatures, GIFBS_HasAVX_NoVLX,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR128RegClassID,
        // (xor:{ *:[v2i64] } VR128:{ *:[v2i64] }:$src1, VR128:{ *:[v2i64] }:$src2)  =>  (VPXORrr:{ *:[v2i64] } VR128:{ *:[v2i64] }:$src1, VR128:{ *:[v2i64] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VPXORrr,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 1553,
        GIR_Done,
      // Label 423: @10866
      GIM_Try, /*On fail goto*//*Label 424*/ 10889, // Rule ID 1555 //
        GIM_CheckFeatures, GIFBS_UseSSE2,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR128RegClassID,
        // (xor:{ *:[v2i64] } VR128:{ *:[v2i64] }:$src1, VR128:{ *:[v2i64] }:$src2)  =>  (PXORrr:{ *:[v2i64] } VR128:{ *:[v2i64] }:$src1, VR128:{ *:[v2i64] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::PXORrr,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 1555,
        GIR_Done,
      // Label 424: @10889
      GIM_Try, /*On fail goto*//*Label 425*/ 10912, // Rule ID 4982 //
        GIM_CheckFeatures, GIFBS_HasAVX512_HasVLX,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128XRegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR128XRegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR128XRegClassID,
        // (xor:{ *:[v2i64] } VR128X:{ *:[v2i64] }:$src1, VR128X:{ *:[v2i64] }:$src2)  =>  (VPXORQZ128rr:{ *:[v2i64] } VR128X:{ *:[v2i64] }:$src1, VR128X:{ *:[v2i64] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VPXORQZ128rr,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 4982,
        GIR_Done,
      // Label 425: @10912
      GIM_Reject,
    // Label 422: @10913
    GIM_Reject,
    // Label 381: @10914
    GIM_Try, /*On fail goto*//*Label 426*/ 11006, // Rule ID 13749 //
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v4s1,
      GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v4s1,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VK1RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VK4RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VK4RegClassID,
      // (xor:{ *:[v4i1] } VK4:{ *:[v4i1] }:$src1, VK4:{ *:[v4i1] }:$src2)  =>  (COPY_TO_REGCLASS:{ *:[v4i1] } (KXORWrr:{ *:[v16i1] } (COPY_TO_REGCLASS:{ *:[v16i1] } VK4:{ *:[v4i1] }:$src1, VK16:{ *:[i32] }), (COPY_TO_REGCLASS:{ *:[v16i1] } VK4:{ *:[v4i1] }:$src2, VK16:{ *:[i32] })), VK1:{ *:[i32] })
      GIR_MakeTempReg, /*TempRegID*/0, /*TypeID*/GILLT_v16s1,
      GIR_MakeTempReg, /*TempRegID*/1, /*TypeID*/GILLT_v16s1,
      GIR_MakeTempReg, /*TempRegID*/2, /*TypeID*/GILLT_v16s1,
      GIR_BuildMI, /*InsnID*/3, /*Opcode*/TargetOpcode::COPY,
      GIR_AddTempRegister, /*InsnID*/3, /*TempRegID*/2, /*TempRegFlags*/RegState::Define,
      GIR_Copy, /*NewInsnID*/3, /*OldInsnID*/0, /*OpIdx*/2, // src2
      GIR_ConstrainSelectedInstOperands, /*InsnID*/3,
      GIR_BuildMI, /*InsnID*/2, /*Opcode*/TargetOpcode::COPY,
      GIR_AddTempRegister, /*InsnID*/2, /*TempRegID*/1, /*TempRegFlags*/RegState::Define,
      GIR_Copy, /*NewInsnID*/2, /*OldInsnID*/0, /*OpIdx*/1, // src1
      GIR_ConstrainSelectedInstOperands, /*InsnID*/2,
      GIR_BuildMI, /*InsnID*/1, /*Opcode*/X86::KXORWrr,
      GIR_AddTempRegister, /*InsnID*/1, /*TempRegID*/0, /*TempRegFlags*/RegState::Define,
      GIR_AddTempRegister, /*InsnID*/1, /*TempRegID*/1, /*TempRegFlags*/0,
      GIR_AddTempRegister, /*InsnID*/1, /*TempRegID*/2, /*TempRegFlags*/0,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/1,
      GIR_BuildMI, /*InsnID*/0, /*Opcode*/TargetOpcode::COPY,
      GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
      GIR_AddTempRegister, /*InsnID*/0, /*TempRegID*/0, /*TempRegFlags*/0,
      GIR_EraseFromParent, /*InsnID*/0,
      GIR_ConstrainOperandRC, /*InsnID*/0, /*Op*/0, /*RC VK1*/8,
      // GIR_Coverage, 13749,
      GIR_Done,
    // Label 426: @11006
    GIM_Reject,
    // Label 382: @11007
    GIM_Try, /*On fail goto*//*Label 427*/ 11087,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v4s64,
      GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v4s64,
      GIM_Try, /*On fail goto*//*Label 428*/ 11040, // Rule ID 1557 //
        GIM_CheckFeatures, GIFBS_HasAVX2_NoVLX,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR256RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR256RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR256RegClassID,
        // (xor:{ *:[v4i64] } VR256:{ *:[v4i64] }:$src1, VR256:{ *:[v4i64] }:$src2)  =>  (VPXORYrr:{ *:[v4i64] } VR256:{ *:[v4i64] }:$src1, VR256:{ *:[v4i64] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VPXORYrr,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 1557,
        GIR_Done,
      // Label 428: @11040
      GIM_Try, /*On fail goto*//*Label 429*/ 11063, // Rule ID 4973 //
        GIM_CheckFeatures, GIFBS_HasAVX512_HasVLX,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR256XRegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR256XRegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR256XRegClassID,
        // (xor:{ *:[v4i64] } VR256X:{ *:[v4i64] }:$src1, VR256X:{ *:[v4i64] }:$src2)  =>  (VPXORQZ256rr:{ *:[v4i64] } VR256X:{ *:[v4i64] }:$src1, VR256X:{ *:[v4i64] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VPXORQZ256rr,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 4973,
        GIR_Done,
      // Label 429: @11063
      GIM_Try, /*On fail goto*//*Label 430*/ 11086, // Rule ID 12517 //
        GIM_CheckFeatures, GIFBS_HasAVX1Only,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR256RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR256RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR256RegClassID,
        // (xor:{ *:[v4i64] } VR256:{ *:[v4i64] }:$src1, VR256:{ *:[v4i64] }:$src2)  =>  (VXORPSYrr:{ *:[v4i64] } VR256:{ *:[v4i64] }:$src1, VR256:{ *:[v4i64] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VXORPSYrr,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 12517,
        GIR_Done,
      // Label 430: @11086
      GIM_Reject,
    // Label 427: @11087
    GIM_Reject,
    // Label 383: @11088
    GIM_Try, /*On fail goto*//*Label 431*/ 11196,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v8s1,
      GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v8s1,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VK8RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VK8RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VK8RegClassID,
      GIM_Try, /*On fail goto*//*Label 432*/ 11121, // Rule ID 3613 //
        GIM_CheckFeatures, GIFBS_HasDQI,
        // (xor:{ *:[v8i1] } VK8:{ *:[v8i1] }:$src1, VK8:{ *:[v8i1] }:$src2)  =>  (KXORBrr:{ *:[v8i1] } VK8:{ *:[v8i1] }:$src1, VK8:{ *:[v8i1] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::KXORBrr,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 3613,
        GIR_Done,
      // Label 432: @11121
      GIM_Try, /*On fail goto*//*Label 433*/ 11195, // Rule ID 13746 //
        GIM_CheckFeatures, GIFBS_NoDQI,
        // (xor:{ *:[v8i1] } VK8:{ *:[v8i1] }:$src1, VK8:{ *:[v8i1] }:$src2)  =>  (COPY_TO_REGCLASS:{ *:[v8i1] } (KXORWrr:{ *:[v16i1] } (COPY_TO_REGCLASS:{ *:[v16i1] } VK8:{ *:[v8i1] }:$src1, VK16:{ *:[i32] }), (COPY_TO_REGCLASS:{ *:[v16i1] } VK8:{ *:[v8i1] }:$src2, VK16:{ *:[i32] })), VK8:{ *:[i32] })
        GIR_MakeTempReg, /*TempRegID*/0, /*TypeID*/GILLT_v16s1,
        GIR_MakeTempReg, /*TempRegID*/1, /*TypeID*/GILLT_v16s1,
        GIR_MakeTempReg, /*TempRegID*/2, /*TypeID*/GILLT_v16s1,
        GIR_BuildMI, /*InsnID*/3, /*Opcode*/TargetOpcode::COPY,
        GIR_AddTempRegister, /*InsnID*/3, /*TempRegID*/2, /*TempRegFlags*/RegState::Define,
        GIR_Copy, /*NewInsnID*/3, /*OldInsnID*/0, /*OpIdx*/2, // src2
        GIR_ConstrainSelectedInstOperands, /*InsnID*/3,
        GIR_BuildMI, /*InsnID*/2, /*Opcode*/TargetOpcode::COPY,
        GIR_AddTempRegister, /*InsnID*/2, /*TempRegID*/1, /*TempRegFlags*/RegState::Define,
        GIR_Copy, /*NewInsnID*/2, /*OldInsnID*/0, /*OpIdx*/1, // src1
        GIR_ConstrainSelectedInstOperands, /*InsnID*/2,
        GIR_BuildMI, /*InsnID*/1, /*Opcode*/X86::KXORWrr,
        GIR_AddTempRegister, /*InsnID*/1, /*TempRegID*/0, /*TempRegFlags*/RegState::Define,
        GIR_AddTempRegister, /*InsnID*/1, /*TempRegID*/1, /*TempRegFlags*/0,
        GIR_AddTempRegister, /*InsnID*/1, /*TempRegID*/2, /*TempRegFlags*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/1,
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/TargetOpcode::COPY,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_AddTempRegister, /*InsnID*/0, /*TempRegID*/0, /*TempRegFlags*/0,
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainOperandRC, /*InsnID*/0, /*Op*/0, /*RC VK8*/12,
        // GIR_Coverage, 13746,
        GIR_Done,
      // Label 433: @11195
      GIM_Reject,
    // Label 431: @11196
    GIM_Reject,
    // Label 384: @11197
    GIM_Try, /*On fail goto*//*Label 434*/ 11228, // Rule ID 4964 //
      GIM_CheckFeatures, GIFBS_HasAVX512,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v8s64,
      GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v8s64,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR512RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR512RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR512RegClassID,
      // (xor:{ *:[v8i64] } VR512:{ *:[v8i64] }:$src1, VR512:{ *:[v8i64] }:$src2)  =>  (VPXORQZrr:{ *:[v8i64] } VR512:{ *:[v8i64] }:$src1, VR512:{ *:[v8i64] }:$src2)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VPXORQZrr,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 4964,
      GIR_Done,
    // Label 434: @11228
    GIM_Reject,
    // Label 385: @11229
    GIM_Try, /*On fail goto*//*Label 435*/ 11260, // Rule ID 3614 //
      GIM_CheckFeatures, GIFBS_HasAVX512,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v16s1,
      GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v16s1,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VK16RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VK16RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VK16RegClassID,
      // (xor:{ *:[v16i1] } VK16:{ *:[v16i1] }:$src1, VK16:{ *:[v16i1] }:$src2)  =>  (KXORWrr:{ *:[v16i1] } VK16:{ *:[v16i1] }:$src1, VK16:{ *:[v16i1] }:$src2)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::KXORWrr,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 3614,
      GIR_Done,
    // Label 435: @11260
    GIM_Reject,
    // Label 386: @11261
    GIM_Try, /*On fail goto*//*Label 436*/ 11292, // Rule ID 3615 //
      GIM_CheckFeatures, GIFBS_HasBWI,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v32s1,
      GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v32s1,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VK32RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VK32RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VK32RegClassID,
      // (xor:{ *:[v32i1] } VK32:{ *:[v32i1] }:$src1, VK32:{ *:[v32i1] }:$src2)  =>  (KXORDrr:{ *:[v32i1] } VK32:{ *:[v32i1] }:$src1, VK32:{ *:[v32i1] }:$src2)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::KXORDrr,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 3615,
      GIR_Done,
    // Label 436: @11292
    GIM_Reject,
    // Label 387: @11293
    GIM_Try, /*On fail goto*//*Label 437*/ 11324, // Rule ID 3616 //
      GIM_CheckFeatures, GIFBS_HasBWI,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v64s1,
      GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v64s1,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VK64RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VK64RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VK64RegClassID,
      // (xor:{ *:[v64i1] } VK64:{ *:[v64i1] }:$src1, VK64:{ *:[v64i1] }:$src2)  =>  (KXORQrr:{ *:[v64i1] } VK64:{ *:[v64i1] }:$src1, VK64:{ *:[v64i1] }:$src2)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::KXORQrr,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 3616,
      GIR_Done,
    // Label 437: @11324
    GIM_Reject,
    // Label 388: @11325
    GIM_Reject,
    // Label 6: @11326
    GIM_SwitchType, /*MI*/0, /*Op*/0, /*[*/3, 24, /*)*//*default:*//*Label 442*/ 11725,
    /*GILLT_s32*//*Label 438*/ 11353,
    /*GILLT_s64*//*Label 439*/ 11515, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    /*GILLT_v32s1*//*Label 440*/ 11677, 0, 0,
    /*GILLT_v64s1*//*Label 441*/ 11701,
    // Label 438: @11353
    GIM_Try, /*On fail goto*//*Label 443*/ 11376, // Rule ID 2194 //
      GIM_CheckFeatures, GIFBS_UseAVX,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_s32,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::FR32RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::GR32RegClassID,
      // (bitconvert:{ *:[f32] } GR32:{ *:[i32] }:$src)  =>  (VMOVDI2SSrr:{ *:[f32] } GR32:{ *:[i32] }:$src)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VMOVDI2SSrr,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 2194,
      GIR_Done,
    // Label 443: @11376
    GIM_Try, /*On fail goto*//*Label 444*/ 11399, // Rule ID 2196 //
      GIM_CheckFeatures, GIFBS_UseSSE2,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_s32,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::FR32RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::GR32RegClassID,
      // (bitconvert:{ *:[f32] } GR32:{ *:[i32] }:$src)  =>  (MOVDI2SSrr:{ *:[f32] } GR32:{ *:[i32] }:$src)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::MOVDI2SSrr,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 2196,
      GIR_Done,
    // Label 444: @11399
    GIM_Try, /*On fail goto*//*Label 445*/ 11422, // Rule ID 2210 //
      GIM_CheckFeatures, GIFBS_UseAVX,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_s32,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::GR32RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::FR32RegClassID,
      // (bitconvert:{ *:[i32] } FR32:{ *:[f32] }:$src)  =>  (VMOVSS2DIrr:{ *:[i32] } FR32:{ *:[f32] }:$src)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VMOVSS2DIrr,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 2210,
      GIR_Done,
    // Label 445: @11422
    GIM_Try, /*On fail goto*//*Label 446*/ 11445, // Rule ID 2212 //
      GIM_CheckFeatures, GIFBS_UseSSE2,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_s32,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::GR32RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::FR32RegClassID,
      // (bitconvert:{ *:[i32] } FR32:{ *:[f32] }:$src)  =>  (MOVSS2DIrr:{ *:[i32] } FR32:{ *:[f32] }:$src)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::MOVSS2DIrr,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 2212,
      GIR_Done,
    // Label 446: @11445
    GIM_Try, /*On fail goto*//*Label 447*/ 11468, // Rule ID 3786 //
      GIM_CheckFeatures, GIFBS_HasAVX512,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_s32,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::FR32XRegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::GR32RegClassID,
      // (bitconvert:{ *:[f32] } GR32:{ *:[i32] }:$src)  =>  (VMOVDI2SSZrr:{ *:[f32] } GR32:{ *:[i32] }:$src)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VMOVDI2SSZrr,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 3786,
      GIR_Done,
    // Label 447: @11468
    GIM_Try, /*On fail goto*//*Label 448*/ 11491, // Rule ID 3792 //
      GIM_CheckFeatures, GIFBS_HasAVX512,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_s32,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::GR32RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::FR32XRegClassID,
      // (bitconvert:{ *:[i32] } FR32X:{ *:[f32] }:$src)  =>  (VMOVSS2DIZrr:{ *:[i32] } FR32X:{ *:[f32] }:$src)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VMOVSS2DIZrr,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 3792,
      GIR_Done,
    // Label 448: @11491
    GIM_Try, /*On fail goto*//*Label 449*/ 11514, // Rule ID 13704 //
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v32s1,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::GR32RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VK32RegClassID,
      // (bitconvert:{ *:[i32] } VK32:{ *:[v32i1] }:$src)  =>  (COPY_TO_REGCLASS:{ *:[i32] } VK32:{ *:[v32i1] }:$src, GR32:{ *:[i32] })
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/TargetOpcode::COPY,
      GIR_ConstrainOperandRC, /*InsnID*/0, /*Op*/0, /*RC GR32*/27,
      // GIR_Coverage, 13704,
      GIR_Done,
    // Label 449: @11514
    GIM_Reject,
    // Label 439: @11515
    GIM_Try, /*On fail goto*//*Label 450*/ 11538, // Rule ID 2189 //
      GIM_CheckFeatures, GIFBS_UseAVX,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_s64,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::FR64RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::GR64RegClassID,
      // (bitconvert:{ *:[f64] } GR64:{ *:[i64] }:$src)  =>  (VMOV64toSDrr:{ *:[f64] } GR64:{ *:[i64] }:$src)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VMOV64toSDrr,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 2189,
      GIR_Done,
    // Label 450: @11538
    GIM_Try, /*On fail goto*//*Label 451*/ 11561, // Rule ID 2193 //
      GIM_CheckFeatures, GIFBS_UseSSE2,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_s64,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::FR64RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::GR64RegClassID,
      // (bitconvert:{ *:[f64] } GR64:{ *:[i64] }:$src)  =>  (MOV64toSDrr:{ *:[f64] } GR64:{ *:[i64] }:$src)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::MOV64toSDrr,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 2193,
      GIR_Done,
    // Label 451: @11561
    GIM_Try, /*On fail goto*//*Label 452*/ 11584, // Rule ID 2205 //
      GIM_CheckFeatures, GIFBS_UseAVX,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_s64,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::GR64RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::FR64RegClassID,
      // (bitconvert:{ *:[i64] } FR64:{ *:[f64] }:$src)  =>  (VMOVSDto64rr:{ *:[i64] } FR64:{ *:[f64] }:$src)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VMOVSDto64rr,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 2205,
      GIR_Done,
    // Label 452: @11584
    GIM_Try, /*On fail goto*//*Label 453*/ 11607, // Rule ID 2208 //
      GIM_CheckFeatures, GIFBS_UseSSE2,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_s64,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::GR64RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::FR64RegClassID,
      // (bitconvert:{ *:[i64] } FR64:{ *:[f64] }:$src)  =>  (MOVSDto64rr:{ *:[i64] } FR64:{ *:[f64] }:$src)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::MOVSDto64rr,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 2208,
      GIR_Done,
    // Label 453: @11607
    GIM_Try, /*On fail goto*//*Label 454*/ 11630, // Rule ID 3782 //
      GIM_CheckFeatures, GIFBS_HasAVX512,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_s64,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::FR64XRegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::GR64RegClassID,
      // (bitconvert:{ *:[f64] } GR64:{ *:[i64] }:$src)  =>  (VMOV64toSDZrr:{ *:[f64] } GR64:{ *:[i64] }:$src)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VMOV64toSDZrr,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 3782,
      GIR_Done,
    // Label 454: @11630
    GIM_Try, /*On fail goto*//*Label 455*/ 11653, // Rule ID 3784 //
      GIM_CheckFeatures, GIFBS_HasAVX512,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_s64,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::GR64RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::FR64XRegClassID,
      // (bitconvert:{ *:[i64] } FR64X:{ *:[f64] }:$src)  =>  (VMOVSDto64Zrr:{ *:[i64] } FR64X:{ *:[f64] }:$src)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VMOVSDto64Zrr,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 3784,
      GIR_Done,
    // Label 455: @11653
    GIM_Try, /*On fail goto*//*Label 456*/ 11676, // Rule ID 13706 //
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v64s1,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::GR64RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VK64RegClassID,
      // (bitconvert:{ *:[i64] } VK64:{ *:[v64i1] }:$src)  =>  (COPY_TO_REGCLASS:{ *:[i64] } VK64:{ *:[v64i1] }:$src, GR64:{ *:[i32] })
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/TargetOpcode::COPY,
      GIR_ConstrainOperandRC, /*InsnID*/0, /*Op*/0, /*RC GR64*/57,
      // GIR_Coverage, 13706,
      GIR_Done,
    // Label 456: @11676
    GIM_Reject,
    // Label 440: @11677
    GIM_Try, /*On fail goto*//*Label 457*/ 11700, // Rule ID 13703 //
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_s32,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VK32RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::GR32RegClassID,
      // (bitconvert:{ *:[v32i1] } GR32:{ *:[i32] }:$src)  =>  (COPY_TO_REGCLASS:{ *:[v32i1] } GR32:{ *:[i32] }:$src, VK32:{ *:[i32] })
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/TargetOpcode::COPY,
      GIR_ConstrainOperandRC, /*InsnID*/0, /*Op*/0, /*RC VK32*/31,
      // GIR_Coverage, 13703,
      GIR_Done,
    // Label 457: @11700
    GIM_Reject,
    // Label 441: @11701
    GIM_Try, /*On fail goto*//*Label 458*/ 11724, // Rule ID 13705 //
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_s64,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VK64RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::GR64RegClassID,
      // (bitconvert:{ *:[v64i1] } GR64:{ *:[i64] }:$src)  =>  (COPY_TO_REGCLASS:{ *:[v64i1] } GR64:{ *:[i64] }:$src, VK64:{ *:[i32] })
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/TargetOpcode::COPY,
      GIR_ConstrainOperandRC, /*InsnID*/0, /*Op*/0, /*RC VK64*/67,
      // GIR_Coverage, 13705,
      GIR_Done,
    // Label 458: @11724
    GIM_Reject,
    // Label 442: @11725
    GIM_Reject,
    // Label 7: @11726
    GIM_Try, /*On fail goto*//*Label 459*/ 12828,
      GIM_CheckNumOperands, /*MI*/0, /*Expected*/3,
      GIM_Try, /*On fail goto*//*Label 460*/ 11771, // Rule ID 1129 //
        GIM_CheckFeatures, GIFBS_HasXOP,
        GIM_CheckIntrinsicID, /*MI*/0, /*Op*/1, Intrinsic::x86_xop_vphsubwd,
        GIM_CheckType, /*MI*/0, /*Op*/0, /*Type*/GILLT_v4s32,
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v8s16,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR128RegClassID,
        // (intrinsic_wo_chain:{ *:[v4i32] } 6645:{ *:[iPTR] }, VR128:{ *:[v8i16] }:$src)  =>  (VPHSUBWDrr:{ *:[v4i32] } VR128:{ *:[v8i16] }:$src)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::VPHSUBWDrr,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/2, // src
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 1129,
        GIR_Done,
      // Label 460: @11771
      GIM_Try, /*On fail goto*//*Label 461*/ 11811, // Rule ID 1131 //
        GIM_CheckFeatures, GIFBS_HasXOP,
        GIM_CheckIntrinsicID, /*MI*/0, /*Op*/1, Intrinsic::x86_xop_vphsubdq,
        GIM_CheckType, /*MI*/0, /*Op*/0, /*Type*/GILLT_v2s64,
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v4s32,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR128RegClassID,
        // (intrinsic_wo_chain:{ *:[v2i64] } 6644:{ *:[iPTR] }, VR128:{ *:[v4i32] }:$src)  =>  (VPHSUBDQrr:{ *:[v2i64] } VR128:{ *:[v4i32] }:$src)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::VPHSUBDQrr,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/2, // src
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 1131,
        GIR_Done,
      // Label 461: @11811
      GIM_Try, /*On fail goto*//*Label 462*/ 11851, // Rule ID 1133 //
        GIM_CheckFeatures, GIFBS_HasXOP,
        GIM_CheckIntrinsicID, /*MI*/0, /*Op*/1, Intrinsic::x86_xop_vphsubbw,
        GIM_CheckType, /*MI*/0, /*Op*/0, /*Type*/GILLT_v8s16,
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v16s8,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR128RegClassID,
        // (intrinsic_wo_chain:{ *:[v8i16] } 6643:{ *:[iPTR] }, VR128:{ *:[v16i8] }:$src)  =>  (VPHSUBBWrr:{ *:[v8i16] } VR128:{ *:[v16i8] }:$src)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::VPHSUBBWrr,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/2, // src
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 1133,
        GIR_Done,
      // Label 462: @11851
      GIM_Try, /*On fail goto*//*Label 463*/ 11891, // Rule ID 1135 //
        GIM_CheckFeatures, GIFBS_HasXOP,
        GIM_CheckIntrinsicID, /*MI*/0, /*Op*/1, Intrinsic::x86_xop_vphaddwq,
        GIM_CheckType, /*MI*/0, /*Op*/0, /*Type*/GILLT_v2s64,
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v8s16,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR128RegClassID,
        // (intrinsic_wo_chain:{ *:[v2i64] } 6642:{ *:[iPTR] }, VR128:{ *:[v8i16] }:$src)  =>  (VPHADDWQrr:{ *:[v2i64] } VR128:{ *:[v8i16] }:$src)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::VPHADDWQrr,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/2, // src
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 1135,
        GIR_Done,
      // Label 463: @11891
      GIM_Try, /*On fail goto*//*Label 464*/ 11931, // Rule ID 1137 //
        GIM_CheckFeatures, GIFBS_HasXOP,
        GIM_CheckIntrinsicID, /*MI*/0, /*Op*/1, Intrinsic::x86_xop_vphaddwd,
        GIM_CheckType, /*MI*/0, /*Op*/0, /*Type*/GILLT_v4s32,
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v8s16,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR128RegClassID,
        // (intrinsic_wo_chain:{ *:[v4i32] } 6641:{ *:[iPTR] }, VR128:{ *:[v8i16] }:$src)  =>  (VPHADDWDrr:{ *:[v4i32] } VR128:{ *:[v8i16] }:$src)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::VPHADDWDrr,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/2, // src
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 1137,
        GIR_Done,
      // Label 464: @11931
      GIM_Try, /*On fail goto*//*Label 465*/ 11971, // Rule ID 1139 //
        GIM_CheckFeatures, GIFBS_HasXOP,
        GIM_CheckIntrinsicID, /*MI*/0, /*Op*/1, Intrinsic::x86_xop_vphadduwq,
        GIM_CheckType, /*MI*/0, /*Op*/0, /*Type*/GILLT_v2s64,
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v8s16,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR128RegClassID,
        // (intrinsic_wo_chain:{ *:[v2i64] } 6640:{ *:[iPTR] }, VR128:{ *:[v8i16] }:$src)  =>  (VPHADDUWQrr:{ *:[v2i64] } VR128:{ *:[v8i16] }:$src)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::VPHADDUWQrr,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/2, // src
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 1139,
        GIR_Done,
      // Label 465: @11971
      GIM_Try, /*On fail goto*//*Label 466*/ 12011, // Rule ID 1141 //
        GIM_CheckFeatures, GIFBS_HasXOP,
        GIM_CheckIntrinsicID, /*MI*/0, /*Op*/1, Intrinsic::x86_xop_vphadduwd,
        GIM_CheckType, /*MI*/0, /*Op*/0, /*Type*/GILLT_v4s32,
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v8s16,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR128RegClassID,
        // (intrinsic_wo_chain:{ *:[v4i32] } 6639:{ *:[iPTR] }, VR128:{ *:[v8i16] }:$src)  =>  (VPHADDUWDrr:{ *:[v4i32] } VR128:{ *:[v8i16] }:$src)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::VPHADDUWDrr,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/2, // src
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 1141,
        GIR_Done,
      // Label 466: @12011
      GIM_Try, /*On fail goto*//*Label 467*/ 12051, // Rule ID 1143 //
        GIM_CheckFeatures, GIFBS_HasXOP,
        GIM_CheckIntrinsicID, /*MI*/0, /*Op*/1, Intrinsic::x86_xop_vphaddudq,
        GIM_CheckType, /*MI*/0, /*Op*/0, /*Type*/GILLT_v2s64,
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v4s32,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR128RegClassID,
        // (intrinsic_wo_chain:{ *:[v2i64] } 6638:{ *:[iPTR] }, VR128:{ *:[v4i32] }:$src)  =>  (VPHADDUDQrr:{ *:[v2i64] } VR128:{ *:[v4i32] }:$src)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::VPHADDUDQrr,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/2, // src
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 1143,
        GIR_Done,
      // Label 467: @12051
      GIM_Try, /*On fail goto*//*Label 468*/ 12091, // Rule ID 1145 //
        GIM_CheckFeatures, GIFBS_HasXOP,
        GIM_CheckIntrinsicID, /*MI*/0, /*Op*/1, Intrinsic::x86_xop_vphaddubw,
        GIM_CheckType, /*MI*/0, /*Op*/0, /*Type*/GILLT_v8s16,
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v16s8,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR128RegClassID,
        // (intrinsic_wo_chain:{ *:[v8i16] } 6637:{ *:[iPTR] }, VR128:{ *:[v16i8] }:$src)  =>  (VPHADDUBWrr:{ *:[v8i16] } VR128:{ *:[v16i8] }:$src)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::VPHADDUBWrr,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/2, // src
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 1145,
        GIR_Done,
      // Label 468: @12091
      GIM_Try, /*On fail goto*//*Label 469*/ 12131, // Rule ID 1147 //
        GIM_CheckFeatures, GIFBS_HasXOP,
        GIM_CheckIntrinsicID, /*MI*/0, /*Op*/1, Intrinsic::x86_xop_vphaddubq,
        GIM_CheckType, /*MI*/0, /*Op*/0, /*Type*/GILLT_v2s64,
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v16s8,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR128RegClassID,
        // (intrinsic_wo_chain:{ *:[v2i64] } 6636:{ *:[iPTR] }, VR128:{ *:[v16i8] }:$src)  =>  (VPHADDUBQrr:{ *:[v2i64] } VR128:{ *:[v16i8] }:$src)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::VPHADDUBQrr,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/2, // src
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 1147,
        GIR_Done,
      // Label 469: @12131
      GIM_Try, /*On fail goto*//*Label 470*/ 12171, // Rule ID 1149 //
        GIM_CheckFeatures, GIFBS_HasXOP,
        GIM_CheckIntrinsicID, /*MI*/0, /*Op*/1, Intrinsic::x86_xop_vphaddubd,
        GIM_CheckType, /*MI*/0, /*Op*/0, /*Type*/GILLT_v4s32,
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v16s8,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR128RegClassID,
        // (intrinsic_wo_chain:{ *:[v4i32] } 6635:{ *:[iPTR] }, VR128:{ *:[v16i8] }:$src)  =>  (VPHADDUBDrr:{ *:[v4i32] } VR128:{ *:[v16i8] }:$src)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::VPHADDUBDrr,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/2, // src
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 1149,
        GIR_Done,
      // Label 470: @12171
      GIM_Try, /*On fail goto*//*Label 471*/ 12211, // Rule ID 1151 //
        GIM_CheckFeatures, GIFBS_HasXOP,
        GIM_CheckIntrinsicID, /*MI*/0, /*Op*/1, Intrinsic::x86_xop_vphadddq,
        GIM_CheckType, /*MI*/0, /*Op*/0, /*Type*/GILLT_v2s64,
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v4s32,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR128RegClassID,
        // (intrinsic_wo_chain:{ *:[v2i64] } 6634:{ *:[iPTR] }, VR128:{ *:[v4i32] }:$src)  =>  (VPHADDDQrr:{ *:[v2i64] } VR128:{ *:[v4i32] }:$src)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::VPHADDDQrr,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/2, // src
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 1151,
        GIR_Done,
      // Label 471: @12211
      GIM_Try, /*On fail goto*//*Label 472*/ 12251, // Rule ID 1153 //
        GIM_CheckFeatures, GIFBS_HasXOP,
        GIM_CheckIntrinsicID, /*MI*/0, /*Op*/1, Intrinsic::x86_xop_vphaddbw,
        GIM_CheckType, /*MI*/0, /*Op*/0, /*Type*/GILLT_v8s16,
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v16s8,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR128RegClassID,
        // (intrinsic_wo_chain:{ *:[v8i16] } 6633:{ *:[iPTR] }, VR128:{ *:[v16i8] }:$src)  =>  (VPHADDBWrr:{ *:[v8i16] } VR128:{ *:[v16i8] }:$src)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::VPHADDBWrr,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/2, // src
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 1153,
        GIR_Done,
      // Label 472: @12251
      GIM_Try, /*On fail goto*//*Label 473*/ 12291, // Rule ID 1155 //
        GIM_CheckFeatures, GIFBS_HasXOP,
        GIM_CheckIntrinsicID, /*MI*/0, /*Op*/1, Intrinsic::x86_xop_vphaddbq,
        GIM_CheckType, /*MI*/0, /*Op*/0, /*Type*/GILLT_v2s64,
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v16s8,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR128RegClassID,
        // (intrinsic_wo_chain:{ *:[v2i64] } 6632:{ *:[iPTR] }, VR128:{ *:[v16i8] }:$src)  =>  (VPHADDBQrr:{ *:[v2i64] } VR128:{ *:[v16i8] }:$src)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::VPHADDBQrr,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/2, // src
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 1155,
        GIR_Done,
      // Label 473: @12291
      GIM_Try, /*On fail goto*//*Label 474*/ 12331, // Rule ID 1157 //
        GIM_CheckFeatures, GIFBS_HasXOP,
        GIM_CheckIntrinsicID, /*MI*/0, /*Op*/1, Intrinsic::x86_xop_vphaddbd,
        GIM_CheckType, /*MI*/0, /*Op*/0, /*Type*/GILLT_v4s32,
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v16s8,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR128RegClassID,
        // (intrinsic_wo_chain:{ *:[v4i32] } 6631:{ *:[iPTR] }, VR128:{ *:[v16i8] }:$src)  =>  (VPHADDBDrr:{ *:[v4i32] } VR128:{ *:[v16i8] }:$src)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::VPHADDBDrr,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/2, // src
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 1157,
        GIR_Done,
      // Label 474: @12331
      GIM_Try, /*On fail goto*//*Label 475*/ 12371, // Rule ID 1159 //
        GIM_CheckFeatures, GIFBS_HasXOP,
        GIM_CheckIntrinsicID, /*MI*/0, /*Op*/1, Intrinsic::x86_xop_vfrcz_ss,
        GIM_CheckType, /*MI*/0, /*Op*/0, /*Type*/GILLT_v4s32,
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v4s32,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR128RegClassID,
        // (intrinsic_wo_chain:{ *:[v4f32] } 6618:{ *:[iPTR] }, VR128:{ *:[v4f32] }:$src)  =>  (VFRCZSSrr:{ *:[v4f32] } VR128:{ *:[v4f32] }:$src)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::VFRCZSSrr,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/2, // src
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 1159,
        GIR_Done,
      // Label 475: @12371
      GIM_Try, /*On fail goto*//*Label 476*/ 12411, // Rule ID 1161 //
        GIM_CheckFeatures, GIFBS_HasXOP,
        GIM_CheckIntrinsicID, /*MI*/0, /*Op*/1, Intrinsic::x86_xop_vfrcz_ps,
        GIM_CheckType, /*MI*/0, /*Op*/0, /*Type*/GILLT_v4s32,
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v4s32,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR128RegClassID,
        // (intrinsic_wo_chain:{ *:[v4f32] } 6615:{ *:[iPTR] }, VR128:{ *:[v4f32] }:$src)  =>  (VFRCZPSrr:{ *:[v4f32] } VR128:{ *:[v4f32] }:$src)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::VFRCZPSrr,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/2, // src
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 1161,
        GIR_Done,
      // Label 476: @12411
      GIM_Try, /*On fail goto*//*Label 477*/ 12451, // Rule ID 1163 //
        GIM_CheckFeatures, GIFBS_HasXOP,
        GIM_CheckIntrinsicID, /*MI*/0, /*Op*/1, Intrinsic::x86_xop_vfrcz_ps_256,
        GIM_CheckType, /*MI*/0, /*Op*/0, /*Type*/GILLT_v8s32,
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v8s32,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR256RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR256RegClassID,
        // (intrinsic_wo_chain:{ *:[v8f32] } 6616:{ *:[iPTR] }, VR256:{ *:[v8f32] }:$src)  =>  (VFRCZPSYrr:{ *:[v8f32] } VR256:{ *:[v8f32] }:$src)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::VFRCZPSYrr,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/2, // src
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 1163,
        GIR_Done,
      // Label 477: @12451
      GIM_Try, /*On fail goto*//*Label 478*/ 12491, // Rule ID 1165 //
        GIM_CheckFeatures, GIFBS_HasXOP,
        GIM_CheckIntrinsicID, /*MI*/0, /*Op*/1, Intrinsic::x86_xop_vfrcz_sd,
        GIM_CheckType, /*MI*/0, /*Op*/0, /*Type*/GILLT_v2s64,
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v2s64,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR128RegClassID,
        // (intrinsic_wo_chain:{ *:[v2f64] } 6617:{ *:[iPTR] }, VR128:{ *:[v2f64] }:$src)  =>  (VFRCZSDrr:{ *:[v2f64] } VR128:{ *:[v2f64] }:$src)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::VFRCZSDrr,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/2, // src
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 1165,
        GIR_Done,
      // Label 478: @12491
      GIM_Try, /*On fail goto*//*Label 479*/ 12531, // Rule ID 1167 //
        GIM_CheckFeatures, GIFBS_HasXOP,
        GIM_CheckIntrinsicID, /*MI*/0, /*Op*/1, Intrinsic::x86_xop_vfrcz_pd,
        GIM_CheckType, /*MI*/0, /*Op*/0, /*Type*/GILLT_v2s64,
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v2s64,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR128RegClassID,
        // (intrinsic_wo_chain:{ *:[v2f64] } 6613:{ *:[iPTR] }, VR128:{ *:[v2f64] }:$src)  =>  (VFRCZPDrr:{ *:[v2f64] } VR128:{ *:[v2f64] }:$src)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::VFRCZPDrr,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/2, // src
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 1167,
        GIR_Done,
      // Label 479: @12531
      GIM_Try, /*On fail goto*//*Label 480*/ 12571, // Rule ID 1169 //
        GIM_CheckFeatures, GIFBS_HasXOP,
        GIM_CheckIntrinsicID, /*MI*/0, /*Op*/1, Intrinsic::x86_xop_vfrcz_pd_256,
        GIM_CheckType, /*MI*/0, /*Op*/0, /*Type*/GILLT_v4s64,
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v4s64,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR256RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR256RegClassID,
        // (intrinsic_wo_chain:{ *:[v4f64] } 6614:{ *:[iPTR] }, VR256:{ *:[v4f64] }:$src)  =>  (VFRCZPDYrr:{ *:[v4f64] } VR256:{ *:[v4f64] }:$src)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::VFRCZPDYrr,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/2, // src
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 1169,
        GIR_Done,
      // Label 480: @12571
      GIM_Try, /*On fail goto*//*Label 481*/ 12611, // Rule ID 2623 //
        GIM_CheckFeatures, GIFBS_HasAES_HasAVX,
        GIM_CheckIntrinsicID, /*MI*/0, /*Op*/1, Intrinsic::x86_aesni_aesimc,
        GIM_CheckType, /*MI*/0, /*Op*/0, /*Type*/GILLT_v2s64,
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v2s64,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR128RegClassID,
        // (intrinsic_wo_chain:{ *:[v2i64] } 5475:{ *:[iPTR] }, VR128:{ *:[v2i64] }:$src1)  =>  (VAESIMCrr:{ *:[v2i64] } VR128:{ *:[v2i64] }:$src1)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::VAESIMCrr,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/2, // src1
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 2623,
        GIR_Done,
      // Label 481: @12611
      GIM_Try, /*On fail goto*//*Label 482*/ 12651, // Rule ID 2625 //
        GIM_CheckFeatures, GIFBS_HasAES_NoAVX,
        GIM_CheckIntrinsicID, /*MI*/0, /*Op*/1, Intrinsic::x86_aesni_aesimc,
        GIM_CheckType, /*MI*/0, /*Op*/0, /*Type*/GILLT_v2s64,
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v2s64,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR128RegClassID,
        // (intrinsic_wo_chain:{ *:[v2i64] } 5475:{ *:[iPTR] }, VR128:{ *:[v2i64] }:$src1)  =>  (AESIMCrr:{ *:[v2i64] } VR128:{ *:[v2i64] }:$src1)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::AESIMCrr,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/2, // src1
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 2625,
        GIR_Done,
      // Label 482: @12651
      GIM_Try, /*On fail goto*//*Label 483*/ 12695, // Rule ID 12563 //
        GIM_CheckFeatures, GIFBS_UseSSE1,
        GIM_CheckIntrinsicID, /*MI*/0, /*Op*/1, Intrinsic::x86_sse_rsqrt_ss,
        GIM_CheckType, /*MI*/0, /*Op*/0, /*Type*/GILLT_v4s32,
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v4s32,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR128RegClassID,
        // (intrinsic_wo_chain:{ *:[v4f32] } 6433:{ *:[iPTR] }, VR128:{ *:[v4f32] }:$src)  =>  (RSQRTSSr_Int:{ *:[v4f32] } VR128:{ *:[v4f32] }:$src, VR128:{ *:[v4f32] }:$src)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::RSQRTSSr_Int,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/2, // src
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/2, // src
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 12563,
        GIR_Done,
      // Label 483: @12695
      GIM_Try, /*On fail goto*//*Label 484*/ 12739, // Rule ID 12565 //
        GIM_CheckFeatures, GIFBS_HasAVX,
        GIM_CheckIntrinsicID, /*MI*/0, /*Op*/1, Intrinsic::x86_sse_rsqrt_ss,
        GIM_CheckType, /*MI*/0, /*Op*/0, /*Type*/GILLT_v4s32,
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v4s32,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR128RegClassID,
        // (intrinsic_wo_chain:{ *:[v4f32] } 6433:{ *:[iPTR] }, VR128:{ *:[v4f32] }:$src)  =>  (VRSQRTSSr_Int:{ *:[v4f32] } VR128:{ *:[v4f32] }:$src, VR128:{ *:[v4f32] }:$src)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::VRSQRTSSr_Int,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/2, // src
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/2, // src
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 12565,
        GIR_Done,
      // Label 484: @12739
      GIM_Try, /*On fail goto*//*Label 485*/ 12783, // Rule ID 12575 //
        GIM_CheckFeatures, GIFBS_UseSSE1,
        GIM_CheckIntrinsicID, /*MI*/0, /*Op*/1, Intrinsic::x86_sse_rcp_ss,
        GIM_CheckType, /*MI*/0, /*Op*/0, /*Type*/GILLT_v4s32,
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v4s32,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR128RegClassID,
        // (intrinsic_wo_chain:{ *:[v4f32] } 6431:{ *:[iPTR] }, VR128:{ *:[v4f32] }:$src)  =>  (RCPSSr_Int:{ *:[v4f32] } VR128:{ *:[v4f32] }:$src, VR128:{ *:[v4f32] }:$src)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::RCPSSr_Int,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/2, // src
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/2, // src
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 12575,
        GIR_Done,
      // Label 485: @12783
      GIM_Try, /*On fail goto*//*Label 486*/ 12827, // Rule ID 12577 //
        GIM_CheckFeatures, GIFBS_HasAVX,
        GIM_CheckIntrinsicID, /*MI*/0, /*Op*/1, Intrinsic::x86_sse_rcp_ss,
        GIM_CheckType, /*MI*/0, /*Op*/0, /*Type*/GILLT_v4s32,
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v4s32,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR128RegClassID,
        // (intrinsic_wo_chain:{ *:[v4f32] } 6431:{ *:[iPTR] }, VR128:{ *:[v4f32] }:$src)  =>  (VRCPSSr_Int:{ *:[v4f32] } VR128:{ *:[v4f32] }:$src, VR128:{ *:[v4f32] }:$src)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::VRCPSSr_Int,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/2, // src
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/2, // src
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 12577,
        GIR_Done,
      // Label 486: @12827
      GIM_Reject,
    // Label 459: @12828
    GIM_Try, /*On fail goto*//*Label 487*/ 15858,
      GIM_CheckNumOperands, /*MI*/0, /*Expected*/4,
      GIM_Try, /*On fail goto*//*Label 488*/ 12889, // Rule ID 2627 //
        GIM_CheckFeatures, GIFBS_HasAES_HasAVX,
        GIM_CheckIntrinsicID, /*MI*/0, /*Op*/1, Intrinsic::x86_aesni_aeskeygenassist,
        GIM_CheckType, /*MI*/0, /*Op*/0, /*Type*/GILLT_v2s64,
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v2s64,
        GIM_CheckType, /*MI*/0, /*Op*/3, /*Type*/GILLT_s8,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR128RegClassID,
        GIM_RecordInsn, /*DefineMI*/1, /*MI*/0, /*OpIdx*/3, // MIs[1]
        GIM_CheckOpcode, /*MI*/1, TargetOpcode::G_CONSTANT,
        // MIs[1] Operand 1
        // No operand predicates
        GIM_CheckIsSafeToFold, /*InsnID*/1,
        // (intrinsic_wo_chain:{ *:[v2i64] } 5476:{ *:[iPTR] }, VR128:{ *:[v2i64] }:$src1, (imm:{ *:[i8] }):$src2)  =>  (VAESKEYGENASSIST128rr:{ *:[v2i64] } VR128:{ *:[v2i64] }:$src1, (imm:{ *:[i8] }):$src2)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::VAESKEYGENASSIST128rr,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/2, // src1
        GIR_CopyConstantAsSImm, /*NewInsnID*/0, /*OldInsnID*/1, // src2
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 2627,
        GIR_Done,
      // Label 488: @12889
      GIM_Try, /*On fail goto*//*Label 489*/ 12945, // Rule ID 2629 //
        GIM_CheckFeatures, GIFBS_HasAES_NoAVX,
        GIM_CheckIntrinsicID, /*MI*/0, /*Op*/1, Intrinsic::x86_aesni_aeskeygenassist,
        GIM_CheckType, /*MI*/0, /*Op*/0, /*Type*/GILLT_v2s64,
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v2s64,
        GIM_CheckType, /*MI*/0, /*Op*/3, /*Type*/GILLT_s8,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR128RegClassID,
        GIM_RecordInsn, /*DefineMI*/1, /*MI*/0, /*OpIdx*/3, // MIs[1]
        GIM_CheckOpcode, /*MI*/1, TargetOpcode::G_CONSTANT,
        // MIs[1] Operand 1
        // No operand predicates
        GIM_CheckIsSafeToFold, /*InsnID*/1,
        // (intrinsic_wo_chain:{ *:[v2i64] } 5476:{ *:[iPTR] }, VR128:{ *:[v2i64] }:$src1, (imm:{ *:[i8] }):$src2)  =>  (AESKEYGENASSIST128rr:{ *:[v2i64] } VR128:{ *:[v2i64] }:$src1, (imm:{ *:[i8] }):$src2)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::AESKEYGENASSIST128rr,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/2, // src1
        GIR_CopyConstantAsSImm, /*NewInsnID*/0, /*OldInsnID*/1, // src2
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 2629,
        GIR_Done,
      // Label 489: @12945
      GIM_Try, /*On fail goto*//*Label 490*/ 12997, // Rule ID 82 //
        GIM_CheckFeatures, GIFBS_HasBMI2,
        GIM_CheckIntrinsicID, /*MI*/0, /*Op*/1, Intrinsic::x86_bmi_pdep_32,
        GIM_CheckType, /*MI*/0, /*Op*/0, /*Type*/GILLT_s32,
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_s32,
        GIM_CheckType, /*MI*/0, /*Op*/3, /*Type*/GILLT_s32,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::GR32RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::GR32RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/3, /*RC*/X86::GR32RegClassID,
        // (intrinsic_wo_chain:{ *:[i32] } 6266:{ *:[iPTR] }, GR32:{ *:[i32] }:$src1, GR32:{ *:[i32] }:$src2)  =>  (PDEP32rr:{ *:[i32] } GR32:{ *:[i32] }:$src1, GR32:{ *:[i32] }:$src2)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::PDEP32rr,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/2, // src1
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/3, // src2
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 82,
        GIR_Done,
      // Label 490: @12997
      GIM_Try, /*On fail goto*//*Label 491*/ 13049, // Rule ID 84 //
        GIM_CheckFeatures, GIFBS_HasBMI2,
        GIM_CheckIntrinsicID, /*MI*/0, /*Op*/1, Intrinsic::x86_bmi_pdep_64,
        GIM_CheckType, /*MI*/0, /*Op*/0, /*Type*/GILLT_s64,
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_s64,
        GIM_CheckType, /*MI*/0, /*Op*/3, /*Type*/GILLT_s64,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::GR64RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::GR64RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/3, /*RC*/X86::GR64RegClassID,
        // (intrinsic_wo_chain:{ *:[i64] } 6267:{ *:[iPTR] }, GR64:{ *:[i64] }:$src1, GR64:{ *:[i64] }:$src2)  =>  (PDEP64rr:{ *:[i64] } GR64:{ *:[i64] }:$src1, GR64:{ *:[i64] }:$src2)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::PDEP64rr,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/2, // src1
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/3, // src2
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 84,
        GIR_Done,
      // Label 491: @13049
      GIM_Try, /*On fail goto*//*Label 492*/ 13101, // Rule ID 86 //
        GIM_CheckFeatures, GIFBS_HasBMI2,
        GIM_CheckIntrinsicID, /*MI*/0, /*Op*/1, Intrinsic::x86_bmi_pext_32,
        GIM_CheckType, /*MI*/0, /*Op*/0, /*Type*/GILLT_s32,
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_s32,
        GIM_CheckType, /*MI*/0, /*Op*/3, /*Type*/GILLT_s32,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::GR32RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::GR32RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/3, /*RC*/X86::GR32RegClassID,
        // (intrinsic_wo_chain:{ *:[i32] } 6268:{ *:[iPTR] }, GR32:{ *:[i32] }:$src1, GR32:{ *:[i32] }:$src2)  =>  (PEXT32rr:{ *:[i32] } GR32:{ *:[i32] }:$src1, GR32:{ *:[i32] }:$src2)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::PEXT32rr,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/2, // src1
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/3, // src2
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 86,
        GIR_Done,
      // Label 492: @13101
      GIM_Try, /*On fail goto*//*Label 493*/ 13153, // Rule ID 88 //
        GIM_CheckFeatures, GIFBS_HasBMI2,
        GIM_CheckIntrinsicID, /*MI*/0, /*Op*/1, Intrinsic::x86_bmi_pext_64,
        GIM_CheckType, /*MI*/0, /*Op*/0, /*Type*/GILLT_s64,
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_s64,
        GIM_CheckType, /*MI*/0, /*Op*/3, /*Type*/GILLT_s64,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::GR64RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::GR64RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/3, /*RC*/X86::GR64RegClassID,
        // (intrinsic_wo_chain:{ *:[i64] } 6269:{ *:[iPTR] }, GR64:{ *:[i64] }:$src1, GR64:{ *:[i64] }:$src2)  =>  (PEXT64rr:{ *:[i64] } GR64:{ *:[i64] }:$src1, GR64:{ *:[i64] }:$src2)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::PEXT64rr,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/2, // src1
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/3, // src2
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 88,
        GIR_Done,
      // Label 493: @13153
      GIM_Try, /*On fail goto*//*Label 494*/ 13205, // Rule ID 1399 //
        GIM_CheckFeatures, GIFBS_HasAVX,
        GIM_CheckIntrinsicID, /*MI*/0, /*Op*/1, Intrinsic::x86_sse2_cvtsd2ss,
        GIM_CheckType, /*MI*/0, /*Op*/0, /*Type*/GILLT_v4s32,
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v4s32,
        GIM_CheckType, /*MI*/0, /*Op*/3, /*Type*/GILLT_v2s64,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/3, /*RC*/X86::VR128RegClassID,
        // (intrinsic_wo_chain:{ *:[v4f32] } 6456:{ *:[iPTR] }, VR128:{ *:[v4f32] }:$src1, VR128:{ *:[v2f64] }:$src2)  =>  (VCVTSD2SSrr_Int:{ *:[v4f32] } VR128:{ *:[v4f32] }:$src1, VR128:{ *:[v2f64] }:$src2)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::VCVTSD2SSrr_Int,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/2, // src1
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/3, // src2
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 1399,
        GIR_Done,
      // Label 494: @13205
      GIM_Try, /*On fail goto*//*Label 495*/ 13257, // Rule ID 1401 //
        GIM_CheckFeatures, GIFBS_UseSSE2,
        GIM_CheckIntrinsicID, /*MI*/0, /*Op*/1, Intrinsic::x86_sse2_cvtsd2ss,
        GIM_CheckType, /*MI*/0, /*Op*/0, /*Type*/GILLT_v4s32,
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v4s32,
        GIM_CheckType, /*MI*/0, /*Op*/3, /*Type*/GILLT_v2s64,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/3, /*RC*/X86::VR128RegClassID,
        // (intrinsic_wo_chain:{ *:[v4f32] } 6456:{ *:[iPTR] }, VR128:{ *:[v4f32] }:$src1, VR128:{ *:[v2f64] }:$src2)  =>  (CVTSD2SSrr_Int:{ *:[v4f32] } VR128:{ *:[v4f32] }:$src1, VR128:{ *:[v2f64] }:$src2)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::CVTSD2SSrr_Int,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/2, // src1
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/3, // src2
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 1401,
        GIR_Done,
      // Label 495: @13257
      GIM_Try, /*On fail goto*//*Label 496*/ 13309, // Rule ID 2309 //
        GIM_CheckFeatures, GIFBS_HasAVX,
        GIM_CheckIntrinsicID, /*MI*/0, /*Op*/1, Intrinsic::x86_ssse3_psign_b_128,
        GIM_CheckType, /*MI*/0, /*Op*/0, /*Type*/GILLT_v16s8,
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v16s8,
        GIM_CheckType, /*MI*/0, /*Op*/3, /*Type*/GILLT_v16s8,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/3, /*RC*/X86::VR128RegClassID,
        // (intrinsic_wo_chain:{ *:[v16i8] } 6573:{ *:[iPTR] }, VR128:{ *:[v16i8] }:$src1, VR128:{ *:[v16i8] }:$src2)  =>  (VPSIGNBrr:{ *:[v16i8] } VR128:{ *:[v16i8] }:$src1, VR128:{ *:[v16i8] }:$src2)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::VPSIGNBrr,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/2, // src1
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/3, // src2
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 2309,
        GIR_Done,
      // Label 496: @13309
      GIM_Try, /*On fail goto*//*Label 497*/ 13361, // Rule ID 2311 //
        GIM_CheckFeatures, GIFBS_HasAVX,
        GIM_CheckIntrinsicID, /*MI*/0, /*Op*/1, Intrinsic::x86_ssse3_psign_w_128,
        GIM_CheckType, /*MI*/0, /*Op*/0, /*Type*/GILLT_v8s16,
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v8s16,
        GIM_CheckType, /*MI*/0, /*Op*/3, /*Type*/GILLT_v8s16,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/3, /*RC*/X86::VR128RegClassID,
        // (intrinsic_wo_chain:{ *:[v8i16] } 6577:{ *:[iPTR] }, VR128:{ *:[v8i16] }:$src1, VR128:{ *:[v8i16] }:$src2)  =>  (VPSIGNWrr:{ *:[v8i16] } VR128:{ *:[v8i16] }:$src1, VR128:{ *:[v8i16] }:$src2)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::VPSIGNWrr,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/2, // src1
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/3, // src2
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 2311,
        GIR_Done,
      // Label 497: @13361
      GIM_Try, /*On fail goto*//*Label 498*/ 13413, // Rule ID 2313 //
        GIM_CheckFeatures, GIFBS_HasAVX,
        GIM_CheckIntrinsicID, /*MI*/0, /*Op*/1, Intrinsic::x86_ssse3_psign_d_128,
        GIM_CheckType, /*MI*/0, /*Op*/0, /*Type*/GILLT_v4s32,
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v4s32,
        GIM_CheckType, /*MI*/0, /*Op*/3, /*Type*/GILLT_v4s32,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/3, /*RC*/X86::VR128RegClassID,
        // (intrinsic_wo_chain:{ *:[v4i32] } 6575:{ *:[iPTR] }, VR128:{ *:[v4i32] }:$src1, VR128:{ *:[v4i32] }:$src2)  =>  (VPSIGNDrr:{ *:[v4i32] } VR128:{ *:[v4i32] }:$src1, VR128:{ *:[v4i32] }:$src2)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::VPSIGNDrr,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/2, // src1
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/3, // src2
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 2313,
        GIR_Done,
      // Label 498: @13413
      GIM_Try, /*On fail goto*//*Label 499*/ 13465, // Rule ID 2315 //
        GIM_CheckFeatures, GIFBS_HasAVX,
        GIM_CheckIntrinsicID, /*MI*/0, /*Op*/1, Intrinsic::x86_ssse3_phadd_sw_128,
        GIM_CheckType, /*MI*/0, /*Op*/0, /*Type*/GILLT_v8s16,
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v8s16,
        GIM_CheckType, /*MI*/0, /*Op*/3, /*Type*/GILLT_v8s16,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/3, /*RC*/X86::VR128RegClassID,
        // (intrinsic_wo_chain:{ *:[v8i16] } 6557:{ *:[iPTR] }, VR128:{ *:[v8i16] }:$src1, VR128:{ *:[v8i16] }:$src2)  =>  (VPHADDSWrr:{ *:[v8i16] } VR128:{ *:[v8i16] }:$src1, VR128:{ *:[v8i16] }:$src2)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::VPHADDSWrr,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/2, // src1
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/3, // src2
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 2315,
        GIR_Done,
      // Label 499: @13465
      GIM_Try, /*On fail goto*//*Label 500*/ 13517, // Rule ID 2317 //
        GIM_CheckFeatures, GIFBS_HasAVX,
        GIM_CheckIntrinsicID, /*MI*/0, /*Op*/1, Intrinsic::x86_ssse3_phsub_sw_128,
        GIM_CheckType, /*MI*/0, /*Op*/0, /*Type*/GILLT_v8s16,
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v8s16,
        GIM_CheckType, /*MI*/0, /*Op*/3, /*Type*/GILLT_v8s16,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/3, /*RC*/X86::VR128RegClassID,
        // (intrinsic_wo_chain:{ *:[v8i16] } 6563:{ *:[iPTR] }, VR128:{ *:[v8i16] }:$src1, VR128:{ *:[v8i16] }:$src2)  =>  (VPHSUBSWrr:{ *:[v8i16] } VR128:{ *:[v8i16] }:$src1, VR128:{ *:[v8i16] }:$src2)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::VPHSUBSWrr,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/2, // src1
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/3, // src2
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 2317,
        GIR_Done,
      // Label 500: @13517
      GIM_Try, /*On fail goto*//*Label 501*/ 13569, // Rule ID 2333 //
        GIM_CheckFeatures, GIFBS_HasAVX2,
        GIM_CheckIntrinsicID, /*MI*/0, /*Op*/1, Intrinsic::x86_avx2_psign_b,
        GIM_CheckType, /*MI*/0, /*Op*/0, /*Type*/GILLT_v32s8,
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v32s8,
        GIM_CheckType, /*MI*/0, /*Op*/3, /*Type*/GILLT_v32s8,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR256RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR256RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/3, /*RC*/X86::VR256RegClassID,
        // (intrinsic_wo_chain:{ *:[v32i8] } 5581:{ *:[iPTR] }, VR256:{ *:[v32i8] }:$src1, VR256:{ *:[v32i8] }:$src2)  =>  (VPSIGNBYrr:{ *:[v32i8] } VR256:{ *:[v32i8] }:$src1, VR256:{ *:[v32i8] }:$src2)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::VPSIGNBYrr,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/2, // src1
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/3, // src2
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 2333,
        GIR_Done,
      // Label 501: @13569
      GIM_Try, /*On fail goto*//*Label 502*/ 13621, // Rule ID 2335 //
        GIM_CheckFeatures, GIFBS_HasAVX2,
        GIM_CheckIntrinsicID, /*MI*/0, /*Op*/1, Intrinsic::x86_avx2_psign_w,
        GIM_CheckType, /*MI*/0, /*Op*/0, /*Type*/GILLT_v16s16,
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v16s16,
        GIM_CheckType, /*MI*/0, /*Op*/3, /*Type*/GILLT_v16s16,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR256RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR256RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/3, /*RC*/X86::VR256RegClassID,
        // (intrinsic_wo_chain:{ *:[v16i16] } 5583:{ *:[iPTR] }, VR256:{ *:[v16i16] }:$src1, VR256:{ *:[v16i16] }:$src2)  =>  (VPSIGNWYrr:{ *:[v16i16] } VR256:{ *:[v16i16] }:$src1, VR256:{ *:[v16i16] }:$src2)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::VPSIGNWYrr,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/2, // src1
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/3, // src2
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 2335,
        GIR_Done,
      // Label 502: @13621
      GIM_Try, /*On fail goto*//*Label 503*/ 13673, // Rule ID 2337 //
        GIM_CheckFeatures, GIFBS_HasAVX2,
        GIM_CheckIntrinsicID, /*MI*/0, /*Op*/1, Intrinsic::x86_avx2_psign_d,
        GIM_CheckType, /*MI*/0, /*Op*/0, /*Type*/GILLT_v8s32,
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v8s32,
        GIM_CheckType, /*MI*/0, /*Op*/3, /*Type*/GILLT_v8s32,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR256RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR256RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/3, /*RC*/X86::VR256RegClassID,
        // (intrinsic_wo_chain:{ *:[v8i32] } 5582:{ *:[iPTR] }, VR256:{ *:[v8i32] }:$src1, VR256:{ *:[v8i32] }:$src2)  =>  (VPSIGNDYrr:{ *:[v8i32] } VR256:{ *:[v8i32] }:$src1, VR256:{ *:[v8i32] }:$src2)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::VPSIGNDYrr,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/2, // src1
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/3, // src2
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 2337,
        GIR_Done,
      // Label 503: @13673
      GIM_Try, /*On fail goto*//*Label 504*/ 13725, // Rule ID 2339 //
        GIM_CheckFeatures, GIFBS_HasAVX2,
        GIM_CheckIntrinsicID, /*MI*/0, /*Op*/1, Intrinsic::x86_avx2_phadd_sw,
        GIM_CheckType, /*MI*/0, /*Op*/0, /*Type*/GILLT_v16s16,
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v16s16,
        GIM_CheckType, /*MI*/0, /*Op*/3, /*Type*/GILLT_v16s16,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR256RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR256RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/3, /*RC*/X86::VR256RegClassID,
        // (intrinsic_wo_chain:{ *:[v16i16] } 5568:{ *:[iPTR] }, VR256:{ *:[v16i16] }:$src1, VR256:{ *:[v16i16] }:$src2)  =>  (VPHADDSWYrr:{ *:[v16i16] } VR256:{ *:[v16i16] }:$src1, VR256:{ *:[v16i16] }:$src2)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::VPHADDSWYrr,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/2, // src1
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/3, // src2
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 2339,
        GIR_Done,
      // Label 504: @13725
      GIM_Try, /*On fail goto*//*Label 505*/ 13777, // Rule ID 2341 //
        GIM_CheckFeatures, GIFBS_HasAVX2,
        GIM_CheckIntrinsicID, /*MI*/0, /*Op*/1, Intrinsic::x86_avx2_phsub_sw,
        GIM_CheckType, /*MI*/0, /*Op*/0, /*Type*/GILLT_v16s16,
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v16s16,
        GIM_CheckType, /*MI*/0, /*Op*/3, /*Type*/GILLT_v16s16,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR256RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR256RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/3, /*RC*/X86::VR256RegClassID,
        // (intrinsic_wo_chain:{ *:[v16i16] } 5571:{ *:[iPTR] }, VR256:{ *:[v16i16] }:$src1, VR256:{ *:[v16i16] }:$src2)  =>  (VPHSUBSWYrr:{ *:[v16i16] } VR256:{ *:[v16i16] }:$src1, VR256:{ *:[v16i16] }:$src2)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::VPHSUBSWYrr,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/2, // src1
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/3, // src2
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 2341,
        GIR_Done,
      // Label 505: @13777
      GIM_Try, /*On fail goto*//*Label 506*/ 13829, // Rule ID 2351 //
        GIM_CheckFeatures, GIFBS_UseSSSE3,
        GIM_CheckIntrinsicID, /*MI*/0, /*Op*/1, Intrinsic::x86_ssse3_psign_b_128,
        GIM_CheckType, /*MI*/0, /*Op*/0, /*Type*/GILLT_v16s8,
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v16s8,
        GIM_CheckType, /*MI*/0, /*Op*/3, /*Type*/GILLT_v16s8,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/3, /*RC*/X86::VR128RegClassID,
        // (intrinsic_wo_chain:{ *:[v16i8] } 6573:{ *:[iPTR] }, VR128:{ *:[v16i8] }:$src1, VR128:{ *:[v16i8] }:$src2)  =>  (PSIGNBrr:{ *:[v16i8] } VR128:{ *:[v16i8] }:$src1, VR128:{ *:[v16i8] }:$src2)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::PSIGNBrr,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/2, // src1
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/3, // src2
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 2351,
        GIR_Done,
      // Label 506: @13829
      GIM_Try, /*On fail goto*//*Label 507*/ 13881, // Rule ID 2353 //
        GIM_CheckFeatures, GIFBS_UseSSSE3,
        GIM_CheckIntrinsicID, /*MI*/0, /*Op*/1, Intrinsic::x86_ssse3_psign_w_128,
        GIM_CheckType, /*MI*/0, /*Op*/0, /*Type*/GILLT_v8s16,
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v8s16,
        GIM_CheckType, /*MI*/0, /*Op*/3, /*Type*/GILLT_v8s16,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/3, /*RC*/X86::VR128RegClassID,
        // (intrinsic_wo_chain:{ *:[v8i16] } 6577:{ *:[iPTR] }, VR128:{ *:[v8i16] }:$src1, VR128:{ *:[v8i16] }:$src2)  =>  (PSIGNWrr:{ *:[v8i16] } VR128:{ *:[v8i16] }:$src1, VR128:{ *:[v8i16] }:$src2)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::PSIGNWrr,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/2, // src1
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/3, // src2
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 2353,
        GIR_Done,
      // Label 507: @13881
      GIM_Try, /*On fail goto*//*Label 508*/ 13933, // Rule ID 2355 //
        GIM_CheckFeatures, GIFBS_UseSSSE3,
        GIM_CheckIntrinsicID, /*MI*/0, /*Op*/1, Intrinsic::x86_ssse3_psign_d_128,
        GIM_CheckType, /*MI*/0, /*Op*/0, /*Type*/GILLT_v4s32,
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v4s32,
        GIM_CheckType, /*MI*/0, /*Op*/3, /*Type*/GILLT_v4s32,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/3, /*RC*/X86::VR128RegClassID,
        // (intrinsic_wo_chain:{ *:[v4i32] } 6575:{ *:[iPTR] }, VR128:{ *:[v4i32] }:$src1, VR128:{ *:[v4i32] }:$src2)  =>  (PSIGNDrr:{ *:[v4i32] } VR128:{ *:[v4i32] }:$src1, VR128:{ *:[v4i32] }:$src2)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::PSIGNDrr,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/2, // src1
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/3, // src2
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 2355,
        GIR_Done,
      // Label 508: @13933
      GIM_Try, /*On fail goto*//*Label 509*/ 13985, // Rule ID 2359 //
        GIM_CheckFeatures, GIFBS_UseSSSE3,
        GIM_CheckIntrinsicID, /*MI*/0, /*Op*/1, Intrinsic::x86_ssse3_phadd_sw_128,
        GIM_CheckType, /*MI*/0, /*Op*/0, /*Type*/GILLT_v8s16,
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v8s16,
        GIM_CheckType, /*MI*/0, /*Op*/3, /*Type*/GILLT_v8s16,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/3, /*RC*/X86::VR128RegClassID,
        // (intrinsic_wo_chain:{ *:[v8i16] } 6557:{ *:[iPTR] }, VR128:{ *:[v8i16] }:$src1, VR128:{ *:[v8i16] }:$src2)  =>  (PHADDSWrr:{ *:[v8i16] } VR128:{ *:[v8i16] }:$src1, VR128:{ *:[v8i16] }:$src2)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::PHADDSWrr,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/2, // src1
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/3, // src2
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 2359,
        GIR_Done,
      // Label 509: @13985
      GIM_Try, /*On fail goto*//*Label 510*/ 14037, // Rule ID 2361 //
        GIM_CheckFeatures, GIFBS_UseSSSE3,
        GIM_CheckIntrinsicID, /*MI*/0, /*Op*/1, Intrinsic::x86_ssse3_phsub_sw_128,
        GIM_CheckType, /*MI*/0, /*Op*/0, /*Type*/GILLT_v8s16,
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v8s16,
        GIM_CheckType, /*MI*/0, /*Op*/3, /*Type*/GILLT_v8s16,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/3, /*RC*/X86::VR128RegClassID,
        // (intrinsic_wo_chain:{ *:[v8i16] } 6563:{ *:[iPTR] }, VR128:{ *:[v8i16] }:$src1, VR128:{ *:[v8i16] }:$src2)  =>  (PHSUBSWrr:{ *:[v8i16] } VR128:{ *:[v8i16] }:$src1, VR128:{ *:[v8i16] }:$src2)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::PHSUBSWrr,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/2, // src1
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/3, // src2
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 2361,
        GIR_Done,
      // Label 510: @14037
      GIM_Try, /*On fail goto*//*Label 511*/ 14089, // Rule ID 2578 //
        GIM_CheckFeatures, GIFBS_HasSSE42,
        GIM_CheckIntrinsicID, /*MI*/0, /*Op*/1, Intrinsic::x86_sse42_crc32_32_8,
        GIM_CheckType, /*MI*/0, /*Op*/0, /*Type*/GILLT_s32,
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_s32,
        GIM_CheckType, /*MI*/0, /*Op*/3, /*Type*/GILLT_s8,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::GR32RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::GR32RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/3, /*RC*/X86::GR8RegClassID,
        // (intrinsic_wo_chain:{ *:[i32] } 6531:{ *:[iPTR] }, GR32:{ *:[i32] }:$src1, GR8:{ *:[i8] }:$src2)  =>  (CRC32r32r8:{ *:[i32] } GR32:{ *:[i32] }:$src1, GR8:{ *:[i8] }:$src2)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::CRC32r32r8,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/2, // src1
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/3, // src2
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 2578,
        GIR_Done,
      // Label 511: @14089
      GIM_Try, /*On fail goto*//*Label 512*/ 14141, // Rule ID 2580 //
        GIM_CheckFeatures, GIFBS_HasSSE42,
        GIM_CheckIntrinsicID, /*MI*/0, /*Op*/1, Intrinsic::x86_sse42_crc32_32_16,
        GIM_CheckType, /*MI*/0, /*Op*/0, /*Type*/GILLT_s32,
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_s32,
        GIM_CheckType, /*MI*/0, /*Op*/3, /*Type*/GILLT_s16,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::GR32RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::GR32RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/3, /*RC*/X86::GR16RegClassID,
        // (intrinsic_wo_chain:{ *:[i32] } 6529:{ *:[iPTR] }, GR32:{ *:[i32] }:$src1, GR16:{ *:[i16] }:$src2)  =>  (CRC32r32r16:{ *:[i32] } GR32:{ *:[i32] }:$src1, GR16:{ *:[i16] }:$src2)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::CRC32r32r16,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/2, // src1
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/3, // src2
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 2580,
        GIR_Done,
      // Label 512: @14141
      GIM_Try, /*On fail goto*//*Label 513*/ 14193, // Rule ID 2582 //
        GIM_CheckFeatures, GIFBS_HasSSE42,
        GIM_CheckIntrinsicID, /*MI*/0, /*Op*/1, Intrinsic::x86_sse42_crc32_32_32,
        GIM_CheckType, /*MI*/0, /*Op*/0, /*Type*/GILLT_s32,
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_s32,
        GIM_CheckType, /*MI*/0, /*Op*/3, /*Type*/GILLT_s32,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::GR32RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::GR32RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/3, /*RC*/X86::GR32RegClassID,
        // (intrinsic_wo_chain:{ *:[i32] } 6530:{ *:[iPTR] }, GR32:{ *:[i32] }:$src1, GR32:{ *:[i32] }:$src2)  =>  (CRC32r32r32:{ *:[i32] } GR32:{ *:[i32] }:$src1, GR32:{ *:[i32] }:$src2)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::CRC32r32r32,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/2, // src1
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/3, // src2
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 2582,
        GIR_Done,
      // Label 513: @14193
      GIM_Try, /*On fail goto*//*Label 514*/ 14245, // Rule ID 2584 //
        GIM_CheckFeatures, GIFBS_HasSSE42,
        GIM_CheckIntrinsicID, /*MI*/0, /*Op*/1, Intrinsic::x86_sse42_crc32_64_64,
        GIM_CheckType, /*MI*/0, /*Op*/0, /*Type*/GILLT_s64,
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_s64,
        GIM_CheckType, /*MI*/0, /*Op*/3, /*Type*/GILLT_s64,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::GR64RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::GR64RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/3, /*RC*/X86::GR64RegClassID,
        // (intrinsic_wo_chain:{ *:[i64] } 6532:{ *:[iPTR] }, GR64:{ *:[i64] }:$src1, GR64:{ *:[i64] }:$src2)  =>  (CRC32r64r64:{ *:[i64] } GR64:{ *:[i64] }:$src1, GR64:{ *:[i64] }:$src2)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::CRC32r64r64,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/2, // src1
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/3, // src2
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 2584,
        GIR_Done,
      // Label 514: @14245
      GIM_Try, /*On fail goto*//*Label 515*/ 14297, // Rule ID 2587 //
        GIM_CheckFeatures, GIFBS_HasSHA,
        GIM_CheckIntrinsicID, /*MI*/0, /*Op*/1, Intrinsic::x86_sha1nexte,
        GIM_CheckType, /*MI*/0, /*Op*/0, /*Type*/GILLT_v4s32,
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v4s32,
        GIM_CheckType, /*MI*/0, /*Op*/3, /*Type*/GILLT_v4s32,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/3, /*RC*/X86::VR128RegClassID,
        // (intrinsic_wo_chain:{ *:[v4i32] } 6399:{ *:[iPTR] }, VR128:{ *:[v4i32] }:$src1, VR128:{ *:[v4i32] }:$src2)  =>  (SHA1NEXTErr:{ *:[v4i32] } VR128:{ *:[v4i32] }:$src1, VR128:{ *:[v4i32] }:$src2)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::SHA1NEXTErr,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/2, // src1
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/3, // src2
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 2587,
        GIR_Done,
      // Label 515: @14297
      GIM_Try, /*On fail goto*//*Label 516*/ 14349, // Rule ID 2589 //
        GIM_CheckFeatures, GIFBS_HasSHA,
        GIM_CheckIntrinsicID, /*MI*/0, /*Op*/1, Intrinsic::x86_sha1msg1,
        GIM_CheckType, /*MI*/0, /*Op*/0, /*Type*/GILLT_v4s32,
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v4s32,
        GIM_CheckType, /*MI*/0, /*Op*/3, /*Type*/GILLT_v4s32,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/3, /*RC*/X86::VR128RegClassID,
        // (intrinsic_wo_chain:{ *:[v4i32] } 6397:{ *:[iPTR] }, VR128:{ *:[v4i32] }:$src1, VR128:{ *:[v4i32] }:$src2)  =>  (SHA1MSG1rr:{ *:[v4i32] } VR128:{ *:[v4i32] }:$src1, VR128:{ *:[v4i32] }:$src2)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::SHA1MSG1rr,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/2, // src1
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/3, // src2
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 2589,
        GIR_Done,
      // Label 516: @14349
      GIM_Try, /*On fail goto*//*Label 517*/ 14401, // Rule ID 2591 //
        GIM_CheckFeatures, GIFBS_HasSHA,
        GIM_CheckIntrinsicID, /*MI*/0, /*Op*/1, Intrinsic::x86_sha1msg2,
        GIM_CheckType, /*MI*/0, /*Op*/0, /*Type*/GILLT_v4s32,
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v4s32,
        GIM_CheckType, /*MI*/0, /*Op*/3, /*Type*/GILLT_v4s32,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/3, /*RC*/X86::VR128RegClassID,
        // (intrinsic_wo_chain:{ *:[v4i32] } 6398:{ *:[iPTR] }, VR128:{ *:[v4i32] }:$src1, VR128:{ *:[v4i32] }:$src2)  =>  (SHA1MSG2rr:{ *:[v4i32] } VR128:{ *:[v4i32] }:$src1, VR128:{ *:[v4i32] }:$src2)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::SHA1MSG2rr,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/2, // src1
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/3, // src2
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 2591,
        GIR_Done,
      // Label 517: @14401
      GIM_Try, /*On fail goto*//*Label 518*/ 14453, // Rule ID 2595 //
        GIM_CheckFeatures, GIFBS_HasSHA,
        GIM_CheckIntrinsicID, /*MI*/0, /*Op*/1, Intrinsic::x86_sha256msg1,
        GIM_CheckType, /*MI*/0, /*Op*/0, /*Type*/GILLT_v4s32,
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v4s32,
        GIM_CheckType, /*MI*/0, /*Op*/3, /*Type*/GILLT_v4s32,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/3, /*RC*/X86::VR128RegClassID,
        // (intrinsic_wo_chain:{ *:[v4i32] } 6401:{ *:[iPTR] }, VR128:{ *:[v4i32] }:$src1, VR128:{ *:[v4i32] }:$src2)  =>  (SHA256MSG1rr:{ *:[v4i32] } VR128:{ *:[v4i32] }:$src1, VR128:{ *:[v4i32] }:$src2)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::SHA256MSG1rr,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/2, // src1
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/3, // src2
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 2595,
        GIR_Done,
      // Label 518: @14453
      GIM_Try, /*On fail goto*//*Label 519*/ 14505, // Rule ID 2597 //
        GIM_CheckFeatures, GIFBS_HasSHA,
        GIM_CheckIntrinsicID, /*MI*/0, /*Op*/1, Intrinsic::x86_sha256msg2,
        GIM_CheckType, /*MI*/0, /*Op*/0, /*Type*/GILLT_v4s32,
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v4s32,
        GIM_CheckType, /*MI*/0, /*Op*/3, /*Type*/GILLT_v4s32,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/3, /*RC*/X86::VR128RegClassID,
        // (intrinsic_wo_chain:{ *:[v4i32] } 6402:{ *:[iPTR] }, VR128:{ *:[v4i32] }:$src1, VR128:{ *:[v4i32] }:$src2)  =>  (SHA256MSG2rr:{ *:[v4i32] } VR128:{ *:[v4i32] }:$src1, VR128:{ *:[v4i32] }:$src2)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::SHA256MSG2rr,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/2, // src1
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/3, // src2
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 2597,
        GIR_Done,
      // Label 519: @14505
      GIM_Try, /*On fail goto*//*Label 520*/ 14557, // Rule ID 2599 //
        GIM_CheckFeatures, GIFBS_HasAES_HasAVX_NoVLX_Or_NoVAES,
        GIM_CheckIntrinsicID, /*MI*/0, /*Op*/1, Intrinsic::x86_aesni_aesenc,
        GIM_CheckType, /*MI*/0, /*Op*/0, /*Type*/GILLT_v2s64,
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v2s64,
        GIM_CheckType, /*MI*/0, /*Op*/3, /*Type*/GILLT_v2s64,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/3, /*RC*/X86::VR128RegClassID,
        // (intrinsic_wo_chain:{ *:[v2i64] } 5469:{ *:[iPTR] }, VR128:{ *:[v2i64] }:$src1, VR128:{ *:[v2i64] }:$src2)  =>  (VAESENCrr:{ *:[v2i64] } VR128:{ *:[v2i64] }:$src1, VR128:{ *:[v2i64] }:$src2)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::VAESENCrr,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/2, // src1
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/3, // src2
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 2599,
        GIR_Done,
      // Label 520: @14557
      GIM_Try, /*On fail goto*//*Label 521*/ 14609, // Rule ID 2601 //
        GIM_CheckFeatures, GIFBS_HasAES_HasAVX_NoVLX_Or_NoVAES,
        GIM_CheckIntrinsicID, /*MI*/0, /*Op*/1, Intrinsic::x86_aesni_aesenclast,
        GIM_CheckType, /*MI*/0, /*Op*/0, /*Type*/GILLT_v2s64,
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v2s64,
        GIM_CheckType, /*MI*/0, /*Op*/3, /*Type*/GILLT_v2s64,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/3, /*RC*/X86::VR128RegClassID,
        // (intrinsic_wo_chain:{ *:[v2i64] } 5472:{ *:[iPTR] }, VR128:{ *:[v2i64] }:$src1, VR128:{ *:[v2i64] }:$src2)  =>  (VAESENCLASTrr:{ *:[v2i64] } VR128:{ *:[v2i64] }:$src1, VR128:{ *:[v2i64] }:$src2)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::VAESENCLASTrr,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/2, // src1
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/3, // src2
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 2601,
        GIR_Done,
      // Label 521: @14609
      GIM_Try, /*On fail goto*//*Label 522*/ 14661, // Rule ID 2603 //
        GIM_CheckFeatures, GIFBS_HasAES_HasAVX_NoVLX_Or_NoVAES,
        GIM_CheckIntrinsicID, /*MI*/0, /*Op*/1, Intrinsic::x86_aesni_aesdec,
        GIM_CheckType, /*MI*/0, /*Op*/0, /*Type*/GILLT_v2s64,
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v2s64,
        GIM_CheckType, /*MI*/0, /*Op*/3, /*Type*/GILLT_v2s64,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/3, /*RC*/X86::VR128RegClassID,
        // (intrinsic_wo_chain:{ *:[v2i64] } 5463:{ *:[iPTR] }, VR128:{ *:[v2i64] }:$src1, VR128:{ *:[v2i64] }:$src2)  =>  (VAESDECrr:{ *:[v2i64] } VR128:{ *:[v2i64] }:$src1, VR128:{ *:[v2i64] }:$src2)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::VAESDECrr,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/2, // src1
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/3, // src2
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 2603,
        GIR_Done,
      // Label 522: @14661
      GIM_Try, /*On fail goto*//*Label 523*/ 14713, // Rule ID 2605 //
        GIM_CheckFeatures, GIFBS_HasAES_HasAVX_NoVLX_Or_NoVAES,
        GIM_CheckIntrinsicID, /*MI*/0, /*Op*/1, Intrinsic::x86_aesni_aesdeclast,
        GIM_CheckType, /*MI*/0, /*Op*/0, /*Type*/GILLT_v2s64,
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v2s64,
        GIM_CheckType, /*MI*/0, /*Op*/3, /*Type*/GILLT_v2s64,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/3, /*RC*/X86::VR128RegClassID,
        // (intrinsic_wo_chain:{ *:[v2i64] } 5466:{ *:[iPTR] }, VR128:{ *:[v2i64] }:$src1, VR128:{ *:[v2i64] }:$src2)  =>  (VAESDECLASTrr:{ *:[v2i64] } VR128:{ *:[v2i64] }:$src1, VR128:{ *:[v2i64] }:$src2)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::VAESDECLASTrr,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/2, // src1
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/3, // src2
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 2605,
        GIR_Done,
      // Label 523: @14713
      GIM_Try, /*On fail goto*//*Label 524*/ 14765, // Rule ID 2607 //
        GIM_CheckFeatures, GIFBS_HasVAES_NoVLX,
        GIM_CheckIntrinsicID, /*MI*/0, /*Op*/1, Intrinsic::x86_aesni_aesenc_256,
        GIM_CheckType, /*MI*/0, /*Op*/0, /*Type*/GILLT_v4s64,
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v4s64,
        GIM_CheckType, /*MI*/0, /*Op*/3, /*Type*/GILLT_v4s64,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR256RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR256RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/3, /*RC*/X86::VR256RegClassID,
        // (intrinsic_wo_chain:{ *:[v4i64] } 5470:{ *:[iPTR] }, VR256:{ *:[v4i64] }:$src1, VR256:{ *:[v4i64] }:$src2)  =>  (VAESENCYrr:{ *:[v4i64] } VR256:{ *:[v4i64] }:$src1, VR256:{ *:[v4i64] }:$src2)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::VAESENCYrr,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/2, // src1
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/3, // src2
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 2607,
        GIR_Done,
      // Label 524: @14765
      GIM_Try, /*On fail goto*//*Label 525*/ 14817, // Rule ID 2609 //
        GIM_CheckFeatures, GIFBS_HasVAES_NoVLX,
        GIM_CheckIntrinsicID, /*MI*/0, /*Op*/1, Intrinsic::x86_aesni_aesenclast_256,
        GIM_CheckType, /*MI*/0, /*Op*/0, /*Type*/GILLT_v4s64,
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v4s64,
        GIM_CheckType, /*MI*/0, /*Op*/3, /*Type*/GILLT_v4s64,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR256RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR256RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/3, /*RC*/X86::VR256RegClassID,
        // (intrinsic_wo_chain:{ *:[v4i64] } 5473:{ *:[iPTR] }, VR256:{ *:[v4i64] }:$src1, VR256:{ *:[v4i64] }:$src2)  =>  (VAESENCLASTYrr:{ *:[v4i64] } VR256:{ *:[v4i64] }:$src1, VR256:{ *:[v4i64] }:$src2)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::VAESENCLASTYrr,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/2, // src1
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/3, // src2
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 2609,
        GIR_Done,
      // Label 525: @14817
      GIM_Try, /*On fail goto*//*Label 526*/ 14869, // Rule ID 2611 //
        GIM_CheckFeatures, GIFBS_HasVAES_NoVLX,
        GIM_CheckIntrinsicID, /*MI*/0, /*Op*/1, Intrinsic::x86_aesni_aesdec_256,
        GIM_CheckType, /*MI*/0, /*Op*/0, /*Type*/GILLT_v4s64,
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v4s64,
        GIM_CheckType, /*MI*/0, /*Op*/3, /*Type*/GILLT_v4s64,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR256RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR256RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/3, /*RC*/X86::VR256RegClassID,
        // (intrinsic_wo_chain:{ *:[v4i64] } 5464:{ *:[iPTR] }, VR256:{ *:[v4i64] }:$src1, VR256:{ *:[v4i64] }:$src2)  =>  (VAESDECYrr:{ *:[v4i64] } VR256:{ *:[v4i64] }:$src1, VR256:{ *:[v4i64] }:$src2)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::VAESDECYrr,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/2, // src1
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/3, // src2
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 2611,
        GIR_Done,
      // Label 526: @14869
      GIM_Try, /*On fail goto*//*Label 527*/ 14921, // Rule ID 2613 //
        GIM_CheckFeatures, GIFBS_HasVAES_NoVLX,
        GIM_CheckIntrinsicID, /*MI*/0, /*Op*/1, Intrinsic::x86_aesni_aesdeclast_256,
        GIM_CheckType, /*MI*/0, /*Op*/0, /*Type*/GILLT_v4s64,
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v4s64,
        GIM_CheckType, /*MI*/0, /*Op*/3, /*Type*/GILLT_v4s64,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR256RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR256RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/3, /*RC*/X86::VR256RegClassID,
        // (intrinsic_wo_chain:{ *:[v4i64] } 5467:{ *:[iPTR] }, VR256:{ *:[v4i64] }:$src1, VR256:{ *:[v4i64] }:$src2)  =>  (VAESDECLASTYrr:{ *:[v4i64] } VR256:{ *:[v4i64] }:$src1, VR256:{ *:[v4i64] }:$src2)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::VAESDECLASTYrr,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/2, // src1
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/3, // src2
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 2613,
        GIR_Done,
      // Label 527: @14921
      GIM_Try, /*On fail goto*//*Label 528*/ 14973, // Rule ID 2615 //
        GIM_CheckFeatures, GIFBS_HasAES_NoAVX,
        GIM_CheckIntrinsicID, /*MI*/0, /*Op*/1, Intrinsic::x86_aesni_aesenc,
        GIM_CheckType, /*MI*/0, /*Op*/0, /*Type*/GILLT_v2s64,
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v2s64,
        GIM_CheckType, /*MI*/0, /*Op*/3, /*Type*/GILLT_v2s64,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/3, /*RC*/X86::VR128RegClassID,
        // (intrinsic_wo_chain:{ *:[v2i64] } 5469:{ *:[iPTR] }, VR128:{ *:[v2i64] }:$src1, VR128:{ *:[v2i64] }:$src2)  =>  (AESENCrr:{ *:[v2i64] } VR128:{ *:[v2i64] }:$src1, VR128:{ *:[v2i64] }:$src2)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::AESENCrr,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/2, // src1
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/3, // src2
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 2615,
        GIR_Done,
      // Label 528: @14973
      GIM_Try, /*On fail goto*//*Label 529*/ 15025, // Rule ID 2617 //
        GIM_CheckFeatures, GIFBS_HasAES_NoAVX,
        GIM_CheckIntrinsicID, /*MI*/0, /*Op*/1, Intrinsic::x86_aesni_aesenclast,
        GIM_CheckType, /*MI*/0, /*Op*/0, /*Type*/GILLT_v2s64,
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v2s64,
        GIM_CheckType, /*MI*/0, /*Op*/3, /*Type*/GILLT_v2s64,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/3, /*RC*/X86::VR128RegClassID,
        // (intrinsic_wo_chain:{ *:[v2i64] } 5472:{ *:[iPTR] }, VR128:{ *:[v2i64] }:$src1, VR128:{ *:[v2i64] }:$src2)  =>  (AESENCLASTrr:{ *:[v2i64] } VR128:{ *:[v2i64] }:$src1, VR128:{ *:[v2i64] }:$src2)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::AESENCLASTrr,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/2, // src1
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/3, // src2
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 2617,
        GIR_Done,
      // Label 529: @15025
      GIM_Try, /*On fail goto*//*Label 530*/ 15077, // Rule ID 2619 //
        GIM_CheckFeatures, GIFBS_HasAES_NoAVX,
        GIM_CheckIntrinsicID, /*MI*/0, /*Op*/1, Intrinsic::x86_aesni_aesdec,
        GIM_CheckType, /*MI*/0, /*Op*/0, /*Type*/GILLT_v2s64,
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v2s64,
        GIM_CheckType, /*MI*/0, /*Op*/3, /*Type*/GILLT_v2s64,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/3, /*RC*/X86::VR128RegClassID,
        // (intrinsic_wo_chain:{ *:[v2i64] } 5463:{ *:[iPTR] }, VR128:{ *:[v2i64] }:$src1, VR128:{ *:[v2i64] }:$src2)  =>  (AESDECrr:{ *:[v2i64] } VR128:{ *:[v2i64] }:$src1, VR128:{ *:[v2i64] }:$src2)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::AESDECrr,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/2, // src1
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/3, // src2
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 2619,
        GIR_Done,
      // Label 530: @15077
      GIM_Try, /*On fail goto*//*Label 531*/ 15129, // Rule ID 2621 //
        GIM_CheckFeatures, GIFBS_HasAES_NoAVX,
        GIM_CheckIntrinsicID, /*MI*/0, /*Op*/1, Intrinsic::x86_aesni_aesdeclast,
        GIM_CheckType, /*MI*/0, /*Op*/0, /*Type*/GILLT_v2s64,
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v2s64,
        GIM_CheckType, /*MI*/0, /*Op*/3, /*Type*/GILLT_v2s64,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/3, /*RC*/X86::VR128RegClassID,
        // (intrinsic_wo_chain:{ *:[v2i64] } 5466:{ *:[iPTR] }, VR128:{ *:[v2i64] }:$src1, VR128:{ *:[v2i64] }:$src2)  =>  (AESDECLASTrr:{ *:[v2i64] } VR128:{ *:[v2i64] }:$src1, VR128:{ *:[v2i64] }:$src2)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::AESDECLASTrr,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/2, // src1
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/3, // src2
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 2621,
        GIR_Done,
      // Label 531: @15129
      GIM_Try, /*On fail goto*//*Label 532*/ 15181, // Rule ID 2638 //
        GIM_CheckFeatures, GIFBS_HasSSE4A,
        GIM_CheckIntrinsicID, /*MI*/0, /*Op*/1, Intrinsic::x86_sse4a_extrq,
        GIM_CheckType, /*MI*/0, /*Op*/0, /*Type*/GILLT_v2s64,
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v2s64,
        GIM_CheckType, /*MI*/0, /*Op*/3, /*Type*/GILLT_v16s8,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/3, /*RC*/X86::VR128RegClassID,
        // (intrinsic_wo_chain:{ *:[v2i64] } 6547:{ *:[iPTR] }, VR128:{ *:[v2i64] }:$src, VR128:{ *:[v16i8] }:$mask)  =>  (EXTRQ:{ *:[v2i64] } VR128:{ *:[v2i64] }:$src, VR128:{ *:[v16i8] }:$mask)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::EXTRQ,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/2, // src
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/3, // mask
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 2638,
        GIR_Done,
      // Label 532: @15181
      GIM_Try, /*On fail goto*//*Label 533*/ 15233, // Rule ID 2640 //
        GIM_CheckFeatures, GIFBS_HasSSE4A,
        GIM_CheckIntrinsicID, /*MI*/0, /*Op*/1, Intrinsic::x86_sse4a_insertq,
        GIM_CheckType, /*MI*/0, /*Op*/0, /*Type*/GILLT_v2s64,
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v2s64,
        GIM_CheckType, /*MI*/0, /*Op*/3, /*Type*/GILLT_v2s64,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/3, /*RC*/X86::VR128RegClassID,
        // (intrinsic_wo_chain:{ *:[v2i64] } 6549:{ *:[iPTR] }, VR128:{ *:[v2i64] }:$src, VR128:{ *:[v2i64] }:$mask)  =>  (INSERTQ:{ *:[v2i64] } VR128:{ *:[v2i64] }:$src, VR128:{ *:[v2i64] }:$mask)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::INSERTQ,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/2, // src
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/3, // mask
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 2640,
        GIR_Done,
      // Label 533: @15233
      GIM_Try, /*On fail goto*//*Label 534*/ 15285, // Rule ID 11162 //
        GIM_CheckFeatures, GIFBS_HasVAES_HasVLX,
        GIM_CheckIntrinsicID, /*MI*/0, /*Op*/1, Intrinsic::x86_aesni_aesenc,
        GIM_CheckType, /*MI*/0, /*Op*/0, /*Type*/GILLT_v2s64,
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v2s64,
        GIM_CheckType, /*MI*/0, /*Op*/3, /*Type*/GILLT_v2s64,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128XRegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR128XRegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/3, /*RC*/X86::VR128XRegClassID,
        // (intrinsic_wo_chain:{ *:[v2i64] } 5469:{ *:[iPTR] }, VR128X:{ *:[v2i64] }:$src1, VR128X:{ *:[v2i64] }:$src2)  =>  (VAESENCZ128rr:{ *:[v2i64] } VR128X:{ *:[v2i64] }:$src1, VR128X:{ *:[v2i64] }:$src2)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::VAESENCZ128rr,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/2, // src1
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/3, // src2
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 11162,
        GIR_Done,
      // Label 534: @15285
      GIM_Try, /*On fail goto*//*Label 535*/ 15337, // Rule ID 11164 //
        GIM_CheckFeatures, GIFBS_HasVAES_HasVLX,
        GIM_CheckIntrinsicID, /*MI*/0, /*Op*/1, Intrinsic::x86_aesni_aesenc_256,
        GIM_CheckType, /*MI*/0, /*Op*/0, /*Type*/GILLT_v4s64,
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v4s64,
        GIM_CheckType, /*MI*/0, /*Op*/3, /*Type*/GILLT_v4s64,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR256XRegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR256XRegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/3, /*RC*/X86::VR256XRegClassID,
        // (intrinsic_wo_chain:{ *:[v4i64] } 5470:{ *:[iPTR] }, VR256X:{ *:[v4i64] }:$src1, VR256X:{ *:[v4i64] }:$src2)  =>  (VAESENCZ256rr:{ *:[v4i64] } VR256X:{ *:[v4i64] }:$src1, VR256X:{ *:[v4i64] }:$src2)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::VAESENCZ256rr,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/2, // src1
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/3, // src2
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 11164,
        GIR_Done,
      // Label 535: @15337
      GIM_Try, /*On fail goto*//*Label 536*/ 15389, // Rule ID 11166 //
        GIM_CheckFeatures, GIFBS_HasAVX512_HasVAES,
        GIM_CheckIntrinsicID, /*MI*/0, /*Op*/1, Intrinsic::x86_aesni_aesenc_512,
        GIM_CheckType, /*MI*/0, /*Op*/0, /*Type*/GILLT_v8s64,
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v8s64,
        GIM_CheckType, /*MI*/0, /*Op*/3, /*Type*/GILLT_v8s64,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR512RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR512RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/3, /*RC*/X86::VR512RegClassID,
        // (intrinsic_wo_chain:{ *:[v8i64] } 5471:{ *:[iPTR] }, VR512:{ *:[v8i64] }:$src1, VR512:{ *:[v8i64] }:$src2)  =>  (VAESENCZrr:{ *:[v8i64] } VR512:{ *:[v8i64] }:$src1, VR512:{ *:[v8i64] }:$src2)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::VAESENCZrr,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/2, // src1
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/3, // src2
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 11166,
        GIR_Done,
      // Label 536: @15389
      GIM_Try, /*On fail goto*//*Label 537*/ 15441, // Rule ID 11168 //
        GIM_CheckFeatures, GIFBS_HasVAES_HasVLX,
        GIM_CheckIntrinsicID, /*MI*/0, /*Op*/1, Intrinsic::x86_aesni_aesenclast,
        GIM_CheckType, /*MI*/0, /*Op*/0, /*Type*/GILLT_v2s64,
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v2s64,
        GIM_CheckType, /*MI*/0, /*Op*/3, /*Type*/GILLT_v2s64,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128XRegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR128XRegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/3, /*RC*/X86::VR128XRegClassID,
        // (intrinsic_wo_chain:{ *:[v2i64] } 5472:{ *:[iPTR] }, VR128X:{ *:[v2i64] }:$src1, VR128X:{ *:[v2i64] }:$src2)  =>  (VAESENCLASTZ128rr:{ *:[v2i64] } VR128X:{ *:[v2i64] }:$src1, VR128X:{ *:[v2i64] }:$src2)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::VAESENCLASTZ128rr,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/2, // src1
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/3, // src2
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 11168,
        GIR_Done,
      // Label 537: @15441
      GIM_Try, /*On fail goto*//*Label 538*/ 15493, // Rule ID 11170 //
        GIM_CheckFeatures, GIFBS_HasVAES_HasVLX,
        GIM_CheckIntrinsicID, /*MI*/0, /*Op*/1, Intrinsic::x86_aesni_aesenclast_256,
        GIM_CheckType, /*MI*/0, /*Op*/0, /*Type*/GILLT_v4s64,
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v4s64,
        GIM_CheckType, /*MI*/0, /*Op*/3, /*Type*/GILLT_v4s64,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR256XRegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR256XRegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/3, /*RC*/X86::VR256XRegClassID,
        // (intrinsic_wo_chain:{ *:[v4i64] } 5473:{ *:[iPTR] }, VR256X:{ *:[v4i64] }:$src1, VR256X:{ *:[v4i64] }:$src2)  =>  (VAESENCLASTZ256rr:{ *:[v4i64] } VR256X:{ *:[v4i64] }:$src1, VR256X:{ *:[v4i64] }:$src2)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::VAESENCLASTZ256rr,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/2, // src1
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/3, // src2
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 11170,
        GIR_Done,
      // Label 538: @15493
      GIM_Try, /*On fail goto*//*Label 539*/ 15545, // Rule ID 11172 //
        GIM_CheckFeatures, GIFBS_HasAVX512_HasVAES,
        GIM_CheckIntrinsicID, /*MI*/0, /*Op*/1, Intrinsic::x86_aesni_aesenclast_512,
        GIM_CheckType, /*MI*/0, /*Op*/0, /*Type*/GILLT_v8s64,
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v8s64,
        GIM_CheckType, /*MI*/0, /*Op*/3, /*Type*/GILLT_v8s64,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR512RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR512RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/3, /*RC*/X86::VR512RegClassID,
        // (intrinsic_wo_chain:{ *:[v8i64] } 5474:{ *:[iPTR] }, VR512:{ *:[v8i64] }:$src1, VR512:{ *:[v8i64] }:$src2)  =>  (VAESENCLASTZrr:{ *:[v8i64] } VR512:{ *:[v8i64] }:$src1, VR512:{ *:[v8i64] }:$src2)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::VAESENCLASTZrr,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/2, // src1
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/3, // src2
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 11172,
        GIR_Done,
      // Label 539: @15545
      GIM_Try, /*On fail goto*//*Label 540*/ 15597, // Rule ID 11174 //
        GIM_CheckFeatures, GIFBS_HasVAES_HasVLX,
        GIM_CheckIntrinsicID, /*MI*/0, /*Op*/1, Intrinsic::x86_aesni_aesdec,
        GIM_CheckType, /*MI*/0, /*Op*/0, /*Type*/GILLT_v2s64,
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v2s64,
        GIM_CheckType, /*MI*/0, /*Op*/3, /*Type*/GILLT_v2s64,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128XRegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR128XRegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/3, /*RC*/X86::VR128XRegClassID,
        // (intrinsic_wo_chain:{ *:[v2i64] } 5463:{ *:[iPTR] }, VR128X:{ *:[v2i64] }:$src1, VR128X:{ *:[v2i64] }:$src2)  =>  (VAESDECZ128rr:{ *:[v2i64] } VR128X:{ *:[v2i64] }:$src1, VR128X:{ *:[v2i64] }:$src2)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::VAESDECZ128rr,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/2, // src1
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/3, // src2
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 11174,
        GIR_Done,
      // Label 540: @15597
      GIM_Try, /*On fail goto*//*Label 541*/ 15649, // Rule ID 11176 //
        GIM_CheckFeatures, GIFBS_HasVAES_HasVLX,
        GIM_CheckIntrinsicID, /*MI*/0, /*Op*/1, Intrinsic::x86_aesni_aesdec_256,
        GIM_CheckType, /*MI*/0, /*Op*/0, /*Type*/GILLT_v4s64,
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v4s64,
        GIM_CheckType, /*MI*/0, /*Op*/3, /*Type*/GILLT_v4s64,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR256XRegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR256XRegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/3, /*RC*/X86::VR256XRegClassID,
        // (intrinsic_wo_chain:{ *:[v4i64] } 5464:{ *:[iPTR] }, VR256X:{ *:[v4i64] }:$src1, VR256X:{ *:[v4i64] }:$src2)  =>  (VAESDECZ256rr:{ *:[v4i64] } VR256X:{ *:[v4i64] }:$src1, VR256X:{ *:[v4i64] }:$src2)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::VAESDECZ256rr,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/2, // src1
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/3, // src2
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 11176,
        GIR_Done,
      // Label 541: @15649
      GIM_Try, /*On fail goto*//*Label 542*/ 15701, // Rule ID 11178 //
        GIM_CheckFeatures, GIFBS_HasAVX512_HasVAES,
        GIM_CheckIntrinsicID, /*MI*/0, /*Op*/1, Intrinsic::x86_aesni_aesdec_512,
        GIM_CheckType, /*MI*/0, /*Op*/0, /*Type*/GILLT_v8s64,
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v8s64,
        GIM_CheckType, /*MI*/0, /*Op*/3, /*Type*/GILLT_v8s64,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR512RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR512RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/3, /*RC*/X86::VR512RegClassID,
        // (intrinsic_wo_chain:{ *:[v8i64] } 5465:{ *:[iPTR] }, VR512:{ *:[v8i64] }:$src1, VR512:{ *:[v8i64] }:$src2)  =>  (VAESDECZrr:{ *:[v8i64] } VR512:{ *:[v8i64] }:$src1, VR512:{ *:[v8i64] }:$src2)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::VAESDECZrr,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/2, // src1
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/3, // src2
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 11178,
        GIR_Done,
      // Label 542: @15701
      GIM_Try, /*On fail goto*//*Label 543*/ 15753, // Rule ID 11180 //
        GIM_CheckFeatures, GIFBS_HasVAES_HasVLX,
        GIM_CheckIntrinsicID, /*MI*/0, /*Op*/1, Intrinsic::x86_aesni_aesdeclast,
        GIM_CheckType, /*MI*/0, /*Op*/0, /*Type*/GILLT_v2s64,
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v2s64,
        GIM_CheckType, /*MI*/0, /*Op*/3, /*Type*/GILLT_v2s64,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128XRegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR128XRegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/3, /*RC*/X86::VR128XRegClassID,
        // (intrinsic_wo_chain:{ *:[v2i64] } 5466:{ *:[iPTR] }, VR128X:{ *:[v2i64] }:$src1, VR128X:{ *:[v2i64] }:$src2)  =>  (VAESDECLASTZ128rr:{ *:[v2i64] } VR128X:{ *:[v2i64] }:$src1, VR128X:{ *:[v2i64] }:$src2)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::VAESDECLASTZ128rr,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/2, // src1
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/3, // src2
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 11180,
        GIR_Done,
      // Label 543: @15753
      GIM_Try, /*On fail goto*//*Label 544*/ 15805, // Rule ID 11182 //
        GIM_CheckFeatures, GIFBS_HasVAES_HasVLX,
        GIM_CheckIntrinsicID, /*MI*/0, /*Op*/1, Intrinsic::x86_aesni_aesdeclast_256,
        GIM_CheckType, /*MI*/0, /*Op*/0, /*Type*/GILLT_v4s64,
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v4s64,
        GIM_CheckType, /*MI*/0, /*Op*/3, /*Type*/GILLT_v4s64,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR256XRegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR256XRegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/3, /*RC*/X86::VR256XRegClassID,
        // (intrinsic_wo_chain:{ *:[v4i64] } 5467:{ *:[iPTR] }, VR256X:{ *:[v4i64] }:$src1, VR256X:{ *:[v4i64] }:$src2)  =>  (VAESDECLASTZ256rr:{ *:[v4i64] } VR256X:{ *:[v4i64] }:$src1, VR256X:{ *:[v4i64] }:$src2)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::VAESDECLASTZ256rr,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/2, // src1
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/3, // src2
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 11182,
        GIR_Done,
      // Label 544: @15805
      GIM_Try, /*On fail goto*//*Label 545*/ 15857, // Rule ID 11184 //
        GIM_CheckFeatures, GIFBS_HasAVX512_HasVAES,
        GIM_CheckIntrinsicID, /*MI*/0, /*Op*/1, Intrinsic::x86_aesni_aesdeclast_512,
        GIM_CheckType, /*MI*/0, /*Op*/0, /*Type*/GILLT_v8s64,
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v8s64,
        GIM_CheckType, /*MI*/0, /*Op*/3, /*Type*/GILLT_v8s64,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR512RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR512RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/3, /*RC*/X86::VR512RegClassID,
        // (intrinsic_wo_chain:{ *:[v8i64] } 5468:{ *:[iPTR] }, VR512:{ *:[v8i64] }:$src1, VR512:{ *:[v8i64] }:$src2)  =>  (VAESDECLASTZrr:{ *:[v8i64] } VR512:{ *:[v8i64] }:$src1, VR512:{ *:[v8i64] }:$src2)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::VAESDECLASTZrr,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/2, // src1
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/3, // src2
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 11184,
        GIR_Done,
      // Label 545: @15857
      GIM_Reject,
    // Label 487: @15858
    GIM_Try, /*On fail goto*//*Label 546*/ 18308,
      GIM_CheckNumOperands, /*MI*/0, /*Expected*/5,
      GIM_Try, /*On fail goto*//*Label 547*/ 15931, // Rule ID 1455 //
        GIM_CheckFeatures, GIFBS_HasAVX,
        GIM_CheckIntrinsicID, /*MI*/0, /*Op*/1, Intrinsic::x86_sse_cmp_ss,
        GIM_CheckType, /*MI*/0, /*Op*/0, /*Type*/GILLT_v4s32,
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v4s32,
        GIM_CheckType, /*MI*/0, /*Op*/3, /*Type*/GILLT_v4s32,
        GIM_CheckType, /*MI*/0, /*Op*/4, /*Type*/GILLT_s8,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/3, /*RC*/X86::VR128RegClassID,
        GIM_RecordInsn, /*DefineMI*/1, /*MI*/0, /*OpIdx*/4, // MIs[1]
        GIM_CheckOpcode, /*MI*/1, TargetOpcode::G_CONSTANT,
        // MIs[1] Operand 1
        // No operand predicates
        GIM_CheckIsSafeToFold, /*InsnID*/1,
        // (intrinsic_wo_chain:{ *:[v4f32] } 6406:{ *:[iPTR] }, VR128:{ *:[v4f32] }:$src1, VR128:{ *:[v4f32] }:$src, (imm:{ *:[i8] }):$cc)  =>  (VCMPSSrr_Int:{ *:[v4f32] } VR128:{ *:[v4f32] }:$src1, VR128:{ *:[v4f32] }:$src, (imm:{ *:[i8] }):$cc)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::VCMPSSrr_Int,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/2, // src1
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/3, // src
        GIR_CopyConstantAsSImm, /*NewInsnID*/0, /*OldInsnID*/1, // cc
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 1455,
        GIR_Done,
      // Label 547: @15931
      GIM_Try, /*On fail goto*//*Label 548*/ 15999, // Rule ID 1457 //
        GIM_CheckFeatures, GIFBS_HasAVX,
        GIM_CheckIntrinsicID, /*MI*/0, /*Op*/1, Intrinsic::x86_sse2_cmp_sd,
        GIM_CheckType, /*MI*/0, /*Op*/0, /*Type*/GILLT_v2s64,
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v2s64,
        GIM_CheckType, /*MI*/0, /*Op*/3, /*Type*/GILLT_v2s64,
        GIM_CheckType, /*MI*/0, /*Op*/4, /*Type*/GILLT_s8,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/3, /*RC*/X86::VR128RegClassID,
        GIM_RecordInsn, /*DefineMI*/1, /*MI*/0, /*OpIdx*/4, // MIs[1]
        GIM_CheckOpcode, /*MI*/1, TargetOpcode::G_CONSTANT,
        // MIs[1] Operand 1
        // No operand predicates
        GIM_CheckIsSafeToFold, /*InsnID*/1,
        // (intrinsic_wo_chain:{ *:[v2f64] } 6444:{ *:[iPTR] }, VR128:{ *:[v2f64] }:$src1, VR128:{ *:[v2f64] }:$src, (imm:{ *:[i8] }):$cc)  =>  (VCMPSDrr_Int:{ *:[v2f64] } VR128:{ *:[v2f64] }:$src1, VR128:{ *:[v2f64] }:$src, (imm:{ *:[i8] }):$cc)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::VCMPSDrr_Int,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/2, // src1
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/3, // src
        GIR_CopyConstantAsSImm, /*NewInsnID*/0, /*OldInsnID*/1, // cc
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 1457,
        GIR_Done,
      // Label 548: @15999
      GIM_Try, /*On fail goto*//*Label 549*/ 16067, // Rule ID 1459 //
        GIM_CheckFeatures, GIFBS_UseSSE1,
        GIM_CheckIntrinsicID, /*MI*/0, /*Op*/1, Intrinsic::x86_sse_cmp_ss,
        GIM_CheckType, /*MI*/0, /*Op*/0, /*Type*/GILLT_v4s32,
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v4s32,
        GIM_CheckType, /*MI*/0, /*Op*/3, /*Type*/GILLT_v4s32,
        GIM_CheckType, /*MI*/0, /*Op*/4, /*Type*/GILLT_s8,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/3, /*RC*/X86::VR128RegClassID,
        GIM_RecordInsn, /*DefineMI*/1, /*MI*/0, /*OpIdx*/4, // MIs[1]
        GIM_CheckOpcode, /*MI*/1, TargetOpcode::G_CONSTANT,
        // MIs[1] Operand 1
        // No operand predicates
        GIM_CheckIsSafeToFold, /*InsnID*/1,
        // (intrinsic_wo_chain:{ *:[v4f32] } 6406:{ *:[iPTR] }, VR128:{ *:[v4f32] }:$src1, VR128:{ *:[v4f32] }:$src, (imm:{ *:[i8] }):$cc)  =>  (CMPSSrr_Int:{ *:[v4f32] } VR128:{ *:[v4f32] }:$src1, VR128:{ *:[v4f32] }:$src, (imm:{ *:[i8] }):$cc)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::CMPSSrr_Int,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/2, // src1
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/3, // src
        GIR_CopyConstantAsSImm, /*NewInsnID*/0, /*OldInsnID*/1, // cc
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 1459,
        GIR_Done,
      // Label 549: @16067
      GIM_Try, /*On fail goto*//*Label 550*/ 16135, // Rule ID 1461 //
        GIM_CheckFeatures, GIFBS_UseSSE2,
        GIM_CheckIntrinsicID, /*MI*/0, /*Op*/1, Intrinsic::x86_sse2_cmp_sd,
        GIM_CheckType, /*MI*/0, /*Op*/0, /*Type*/GILLT_v2s64,
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v2s64,
        GIM_CheckType, /*MI*/0, /*Op*/3, /*Type*/GILLT_v2s64,
        GIM_CheckType, /*MI*/0, /*Op*/4, /*Type*/GILLT_s8,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/3, /*RC*/X86::VR128RegClassID,
        GIM_RecordInsn, /*DefineMI*/1, /*MI*/0, /*OpIdx*/4, // MIs[1]
        GIM_CheckOpcode, /*MI*/1, TargetOpcode::G_CONSTANT,
        // MIs[1] Operand 1
        // No operand predicates
        GIM_CheckIsSafeToFold, /*InsnID*/1,
        // (intrinsic_wo_chain:{ *:[v2f64] } 6444:{ *:[iPTR] }, VR128:{ *:[v2f64] }:$src1, VR128:{ *:[v2f64] }:$src, (imm:{ *:[i8] }):$cc)  =>  (CMPSDrr_Int:{ *:[v2f64] } VR128:{ *:[v2f64] }:$src1, VR128:{ *:[v2f64] }:$src, (imm:{ *:[i8] }):$cc)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::CMPSDrr_Int,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/2, // src1
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/3, // src
        GIR_CopyConstantAsSImm, /*NewInsnID*/0, /*OldInsnID*/1, // cc
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 1461,
        GIR_Done,
      // Label 550: @16135
      GIM_Try, /*On fail goto*//*Label 551*/ 16203, // Rule ID 2519 //
        GIM_CheckFeatures, GIFBS_HasAVX,
        GIM_CheckIntrinsicID, /*MI*/0, /*Op*/1, Intrinsic::x86_sse41_mpsadbw,
        GIM_CheckType, /*MI*/0, /*Op*/0, /*Type*/GILLT_v8s16,
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v16s8,
        GIM_CheckType, /*MI*/0, /*Op*/3, /*Type*/GILLT_v16s8,
        GIM_CheckType, /*MI*/0, /*Op*/4, /*Type*/GILLT_s8,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/3, /*RC*/X86::VR128RegClassID,
        GIM_RecordInsn, /*DefineMI*/1, /*MI*/0, /*OpIdx*/4, // MIs[1]
        GIM_CheckOpcode, /*MI*/1, TargetOpcode::G_CONSTANT,
        // MIs[1] Operand 1
        // No operand predicates
        GIM_CheckIsSafeToFold, /*InsnID*/1,
        // (intrinsic_wo_chain:{ *:[v8i16] } 6518:{ *:[iPTR] }, VR128:{ *:[v16i8] }:$src1, VR128:{ *:[v16i8] }:$src2, (imm:{ *:[i8] }):$src3)  =>  (VMPSADBWrri:{ *:[v8i16] } VR128:{ *:[v16i8] }:$src1, VR128:{ *:[v16i8] }:$src2, (imm:{ *:[i8] }):$src3)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::VMPSADBWrri,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/2, // src1
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/3, // src2
        GIR_CopyConstantAsSImm, /*NewInsnID*/0, /*OldInsnID*/1, // src3
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 2519,
        GIR_Done,
      // Label 551: @16203
      GIM_Try, /*On fail goto*//*Label 552*/ 16271, // Rule ID 2521 //
        GIM_CheckFeatures, GIFBS_HasAVX,
        GIM_CheckIntrinsicID, /*MI*/0, /*Op*/1, Intrinsic::x86_sse41_dpps,
        GIM_CheckType, /*MI*/0, /*Op*/0, /*Type*/GILLT_v4s32,
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v4s32,
        GIM_CheckType, /*MI*/0, /*Op*/3, /*Type*/GILLT_v4s32,
        GIM_CheckType, /*MI*/0, /*Op*/4, /*Type*/GILLT_s8,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/3, /*RC*/X86::VR128RegClassID,
        GIM_RecordInsn, /*DefineMI*/1, /*MI*/0, /*OpIdx*/4, // MIs[1]
        GIM_CheckOpcode, /*MI*/1, TargetOpcode::G_CONSTANT,
        // MIs[1] Operand 1
        // No operand predicates
        GIM_CheckIsSafeToFold, /*InsnID*/1,
        // (intrinsic_wo_chain:{ *:[v4f32] } 6516:{ *:[iPTR] }, VR128:{ *:[v4f32] }:$src1, VR128:{ *:[v4f32] }:$src2, (imm:{ *:[i8] }):$src3)  =>  (VDPPSrri:{ *:[v4f32] } VR128:{ *:[v4f32] }:$src1, VR128:{ *:[v4f32] }:$src2, (imm:{ *:[i8] }):$src3)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::VDPPSrri,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/2, // src1
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/3, // src2
        GIR_CopyConstantAsSImm, /*NewInsnID*/0, /*OldInsnID*/1, // src3
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 2521,
        GIR_Done,
      // Label 552: @16271
      GIM_Try, /*On fail goto*//*Label 553*/ 16339, // Rule ID 2523 //
        GIM_CheckFeatures, GIFBS_HasAVX,
        GIM_CheckIntrinsicID, /*MI*/0, /*Op*/1, Intrinsic::x86_sse41_dppd,
        GIM_CheckType, /*MI*/0, /*Op*/0, /*Type*/GILLT_v2s64,
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v2s64,
        GIM_CheckType, /*MI*/0, /*Op*/3, /*Type*/GILLT_v2s64,
        GIM_CheckType, /*MI*/0, /*Op*/4, /*Type*/GILLT_s8,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/3, /*RC*/X86::VR128RegClassID,
        GIM_RecordInsn, /*DefineMI*/1, /*MI*/0, /*OpIdx*/4, // MIs[1]
        GIM_CheckOpcode, /*MI*/1, TargetOpcode::G_CONSTANT,
        // MIs[1] Operand 1
        // No operand predicates
        GIM_CheckIsSafeToFold, /*InsnID*/1,
        // (intrinsic_wo_chain:{ *:[v2f64] } 6515:{ *:[iPTR] }, VR128:{ *:[v2f64] }:$src1, VR128:{ *:[v2f64] }:$src2, (imm:{ *:[i8] }):$src3)  =>  (VDPPDrri:{ *:[v2f64] } VR128:{ *:[v2f64] }:$src1, VR128:{ *:[v2f64] }:$src2, (imm:{ *:[i8] }):$src3)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::VDPPDrri,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/2, // src1
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/3, // src2
        GIR_CopyConstantAsSImm, /*NewInsnID*/0, /*OldInsnID*/1, // src3
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 2523,
        GIR_Done,
      // Label 553: @16339
      GIM_Try, /*On fail goto*//*Label 554*/ 16407, // Rule ID 2525 //
        GIM_CheckFeatures, GIFBS_HasAVX,
        GIM_CheckIntrinsicID, /*MI*/0, /*Op*/1, Intrinsic::x86_avx_dp_ps_256,
        GIM_CheckType, /*MI*/0, /*Op*/0, /*Type*/GILLT_v8s32,
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v8s32,
        GIM_CheckType, /*MI*/0, /*Op*/3, /*Type*/GILLT_v8s32,
        GIM_CheckType, /*MI*/0, /*Op*/4, /*Type*/GILLT_s8,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR256RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR256RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/3, /*RC*/X86::VR256RegClassID,
        GIM_RecordInsn, /*DefineMI*/1, /*MI*/0, /*OpIdx*/4, // MIs[1]
        GIM_CheckOpcode, /*MI*/1, TargetOpcode::G_CONSTANT,
        // MIs[1] Operand 1
        // No operand predicates
        GIM_CheckIsSafeToFold, /*InsnID*/1,
        // (intrinsic_wo_chain:{ *:[v8f32] } 5488:{ *:[iPTR] }, VR256:{ *:[v8f32] }:$src1, VR256:{ *:[v8f32] }:$src2, (imm:{ *:[i8] }):$src3)  =>  (VDPPSYrri:{ *:[v8f32] } VR256:{ *:[v8f32] }:$src1, VR256:{ *:[v8f32] }:$src2, (imm:{ *:[i8] }):$src3)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::VDPPSYrri,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/2, // src1
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/3, // src2
        GIR_CopyConstantAsSImm, /*NewInsnID*/0, /*OldInsnID*/1, // src3
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 2525,
        GIR_Done,
      // Label 554: @16407
      GIM_Try, /*On fail goto*//*Label 555*/ 16475, // Rule ID 2527 //
        GIM_CheckFeatures, GIFBS_HasAVX2,
        GIM_CheckIntrinsicID, /*MI*/0, /*Op*/1, Intrinsic::x86_avx2_mpsadbw,
        GIM_CheckType, /*MI*/0, /*Op*/0, /*Type*/GILLT_v16s16,
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v32s8,
        GIM_CheckType, /*MI*/0, /*Op*/3, /*Type*/GILLT_v32s8,
        GIM_CheckType, /*MI*/0, /*Op*/4, /*Type*/GILLT_s8,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR256RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR256RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/3, /*RC*/X86::VR256RegClassID,
        GIM_RecordInsn, /*DefineMI*/1, /*MI*/0, /*OpIdx*/4, // MIs[1]
        GIM_CheckOpcode, /*MI*/1, TargetOpcode::G_CONSTANT,
        // MIs[1] Operand 1
        // No operand predicates
        GIM_CheckIsSafeToFold, /*InsnID*/1,
        // (intrinsic_wo_chain:{ *:[v16i16] } 5557:{ *:[iPTR] }, VR256:{ *:[v32i8] }:$src1, VR256:{ *:[v32i8] }:$src2, (imm:{ *:[i8] }):$src3)  =>  (VMPSADBWYrri:{ *:[v16i16] } VR256:{ *:[v32i8] }:$src1, VR256:{ *:[v32i8] }:$src2, (imm:{ *:[i8] }):$src3)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::VMPSADBWYrri,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/2, // src1
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/3, // src2
        GIR_CopyConstantAsSImm, /*NewInsnID*/0, /*OldInsnID*/1, // src3
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 2527,
        GIR_Done,
      // Label 555: @16475
      GIM_Try, /*On fail goto*//*Label 556*/ 16543, // Rule ID 2529 //
        GIM_CheckFeatures, GIFBS_UseSSE41,
        GIM_CheckIntrinsicID, /*MI*/0, /*Op*/1, Intrinsic::x86_sse41_mpsadbw,
        GIM_CheckType, /*MI*/0, /*Op*/0, /*Type*/GILLT_v8s16,
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v16s8,
        GIM_CheckType, /*MI*/0, /*Op*/3, /*Type*/GILLT_v16s8,
        GIM_CheckType, /*MI*/0, /*Op*/4, /*Type*/GILLT_s8,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/3, /*RC*/X86::VR128RegClassID,
        GIM_RecordInsn, /*DefineMI*/1, /*MI*/0, /*OpIdx*/4, // MIs[1]
        GIM_CheckOpcode, /*MI*/1, TargetOpcode::G_CONSTANT,
        // MIs[1] Operand 1
        // No operand predicates
        GIM_CheckIsSafeToFold, /*InsnID*/1,
        // (intrinsic_wo_chain:{ *:[v8i16] } 6518:{ *:[iPTR] }, VR128:{ *:[v16i8] }:$src1, VR128:{ *:[v16i8] }:$src2, (imm:{ *:[i8] }):$src3)  =>  (MPSADBWrri:{ *:[v8i16] } VR128:{ *:[v16i8] }:$src1, VR128:{ *:[v16i8] }:$src2, (imm:{ *:[i8] }):$src3)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::MPSADBWrri,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/2, // src1
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/3, // src2
        GIR_CopyConstantAsSImm, /*NewInsnID*/0, /*OldInsnID*/1, // src3
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 2529,
        GIR_Done,
      // Label 556: @16543
      GIM_Try, /*On fail goto*//*Label 557*/ 16611, // Rule ID 2531 //
        GIM_CheckFeatures, GIFBS_UseSSE41,
        GIM_CheckIntrinsicID, /*MI*/0, /*Op*/1, Intrinsic::x86_sse41_dpps,
        GIM_CheckType, /*MI*/0, /*Op*/0, /*Type*/GILLT_v4s32,
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v4s32,
        GIM_CheckType, /*MI*/0, /*Op*/3, /*Type*/GILLT_v4s32,
        GIM_CheckType, /*MI*/0, /*Op*/4, /*Type*/GILLT_s8,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/3, /*RC*/X86::VR128RegClassID,
        GIM_RecordInsn, /*DefineMI*/1, /*MI*/0, /*OpIdx*/4, // MIs[1]
        GIM_CheckOpcode, /*MI*/1, TargetOpcode::G_CONSTANT,
        // MIs[1] Operand 1
        // No operand predicates
        GIM_CheckIsSafeToFold, /*InsnID*/1,
        // (intrinsic_wo_chain:{ *:[v4f32] } 6516:{ *:[iPTR] }, VR128:{ *:[v4f32] }:$src1, VR128:{ *:[v4f32] }:$src2, (imm:{ *:[i8] }):$src3)  =>  (DPPSrri:{ *:[v4f32] } VR128:{ *:[v4f32] }:$src1, VR128:{ *:[v4f32] }:$src2, (imm:{ *:[i8] }):$src3)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::DPPSrri,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/2, // src1
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/3, // src2
        GIR_CopyConstantAsSImm, /*NewInsnID*/0, /*OldInsnID*/1, // src3
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 2531,
        GIR_Done,
      // Label 557: @16611
      GIM_Try, /*On fail goto*//*Label 558*/ 16679, // Rule ID 2533 //
        GIM_CheckFeatures, GIFBS_UseSSE41,
        GIM_CheckIntrinsicID, /*MI*/0, /*Op*/1, Intrinsic::x86_sse41_dppd,
        GIM_CheckType, /*MI*/0, /*Op*/0, /*Type*/GILLT_v2s64,
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v2s64,
        GIM_CheckType, /*MI*/0, /*Op*/3, /*Type*/GILLT_v2s64,
        GIM_CheckType, /*MI*/0, /*Op*/4, /*Type*/GILLT_s8,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/3, /*RC*/X86::VR128RegClassID,
        GIM_RecordInsn, /*DefineMI*/1, /*MI*/0, /*OpIdx*/4, // MIs[1]
        GIM_CheckOpcode, /*MI*/1, TargetOpcode::G_CONSTANT,
        // MIs[1] Operand 1
        // No operand predicates
        GIM_CheckIsSafeToFold, /*InsnID*/1,
        // (intrinsic_wo_chain:{ *:[v2f64] } 6515:{ *:[iPTR] }, VR128:{ *:[v2f64] }:$src1, VR128:{ *:[v2f64] }:$src2, (imm:{ *:[i8] }):$src3)  =>  (DPPDrri:{ *:[v2f64] } VR128:{ *:[v2f64] }:$src1, VR128:{ *:[v2f64] }:$src2, (imm:{ *:[i8] }):$src3)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::DPPDrri,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/2, // src1
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/3, // src2
        GIR_CopyConstantAsSImm, /*NewInsnID*/0, /*OldInsnID*/1, // src3
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 2533,
        GIR_Done,
      // Label 558: @16679
      GIM_Try, /*On fail goto*//*Label 559*/ 16747, // Rule ID 2585 //
        GIM_CheckFeatures, GIFBS_HasSHA,
        GIM_CheckIntrinsicID, /*MI*/0, /*Op*/1, Intrinsic::x86_sha1rnds4,
        GIM_CheckType, /*MI*/0, /*Op*/0, /*Type*/GILLT_v4s32,
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v4s32,
        GIM_CheckType, /*MI*/0, /*Op*/3, /*Type*/GILLT_v4s32,
        GIM_CheckType, /*MI*/0, /*Op*/4, /*Type*/GILLT_s8,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/3, /*RC*/X86::VR128RegClassID,
        GIM_RecordInsn, /*DefineMI*/1, /*MI*/0, /*OpIdx*/4, // MIs[1]
        GIM_CheckOpcode, /*MI*/1, TargetOpcode::G_CONSTANT,
        // MIs[1] Operand 1
        // No operand predicates
        GIM_CheckIsSafeToFold, /*InsnID*/1,
        // (intrinsic_wo_chain:{ *:[v4i32] } 6400:{ *:[iPTR] }, VR128:{ *:[v4i32] }:$src1, VR128:{ *:[v4i32] }:$src2, (imm:{ *:[i8] }):$src3)  =>  (SHA1RNDS4rri:{ *:[v4i32] } VR128:{ *:[v4i32] }:$src1, VR128:{ *:[v4i32] }:$src2, (imm:{ *:[i8] }):$src3)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::SHA1RNDS4rri,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/2, // src1
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/3, // src2
        GIR_CopyConstantAsSImm, /*NewInsnID*/0, /*OldInsnID*/1, // src3
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 2585,
        GIR_Done,
      // Label 559: @16747
      GIM_Try, /*On fail goto*//*Label 560*/ 16815, // Rule ID 2631 //
        GIM_CheckFeatures, GIFBS_HasPCLMUL_NoAVX,
        GIM_CheckIntrinsicID, /*MI*/0, /*Op*/1, Intrinsic::x86_pclmulqdq,
        GIM_CheckType, /*MI*/0, /*Op*/0, /*Type*/GILLT_v2s64,
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v2s64,
        GIM_CheckType, /*MI*/0, /*Op*/3, /*Type*/GILLT_v2s64,
        GIM_CheckType, /*MI*/0, /*Op*/4, /*Type*/GILLT_s8,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/3, /*RC*/X86::VR128RegClassID,
        GIM_RecordInsn, /*DefineMI*/1, /*MI*/0, /*OpIdx*/4, // MIs[1]
        GIM_CheckOpcode, /*MI*/1, TargetOpcode::G_CONSTANT,
        // MIs[1] Operand 1
        // No operand predicates
        GIM_CheckIsSafeToFold, /*InsnID*/1,
        // (intrinsic_wo_chain:{ *:[v2i64] } 6368:{ *:[iPTR] }, VR128:{ *:[v2i64] }:$src1, VR128:{ *:[v2i64] }:$src2, (imm:{ *:[i8] }):$src3)  =>  (PCLMULQDQrr:{ *:[v2i64] } VR128:{ *:[v2i64] }:$src1, VR128:{ *:[v2i64] }:$src2, (imm:{ *:[i8] }):$src3)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::PCLMULQDQrr,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/2, // src1
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/3, // src2
        GIR_CopyConstantAsSImm, /*NewInsnID*/0, /*OldInsnID*/1, // src3
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 2631,
        GIR_Done,
      // Label 560: @16815
      GIM_Try, /*On fail goto*//*Label 561*/ 16883, // Rule ID 2633 //
        GIM_CheckFeatures, GIFBS_HasAVX_HasPCLMUL_NoVLX_Or_NoVPCLMULQDQ,
        GIM_CheckIntrinsicID, /*MI*/0, /*Op*/1, Intrinsic::x86_pclmulqdq,
        GIM_CheckType, /*MI*/0, /*Op*/0, /*Type*/GILLT_v2s64,
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v2s64,
        GIM_CheckType, /*MI*/0, /*Op*/3, /*Type*/GILLT_v2s64,
        GIM_CheckType, /*MI*/0, /*Op*/4, /*Type*/GILLT_s8,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/3, /*RC*/X86::VR128RegClassID,
        GIM_RecordInsn, /*DefineMI*/1, /*MI*/0, /*OpIdx*/4, // MIs[1]
        GIM_CheckOpcode, /*MI*/1, TargetOpcode::G_CONSTANT,
        // MIs[1] Operand 1
        // No operand predicates
        GIM_CheckIsSafeToFold, /*InsnID*/1,
        // (intrinsic_wo_chain:{ *:[v2i64] } 6368:{ *:[iPTR] }, VR128:{ *:[v2i64] }:$src1, VR128:{ *:[v2i64] }:$src2, (imm:{ *:[i8] }):$src3)  =>  (VPCLMULQDQrr:{ *:[v2i64] } VR128:{ *:[v2i64] }:$src1, VR128:{ *:[v2i64] }:$src2, (imm:{ *:[i8] }):$src3)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::VPCLMULQDQrr,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/2, // src1
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/3, // src2
        GIR_CopyConstantAsSImm, /*NewInsnID*/0, /*OldInsnID*/1, // src3
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 2633,
        GIR_Done,
      // Label 561: @16883
      GIM_Try, /*On fail goto*//*Label 562*/ 16951, // Rule ID 2635 //
        GIM_CheckFeatures, GIFBS_HasVPCLMULQDQ_NoVLX,
        GIM_CheckIntrinsicID, /*MI*/0, /*Op*/1, Intrinsic::x86_pclmulqdq_256,
        GIM_CheckType, /*MI*/0, /*Op*/0, /*Type*/GILLT_v4s64,
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v4s64,
        GIM_CheckType, /*MI*/0, /*Op*/3, /*Type*/GILLT_v4s64,
        GIM_CheckType, /*MI*/0, /*Op*/4, /*Type*/GILLT_s8,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR256RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR256RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/3, /*RC*/X86::VR256RegClassID,
        GIM_RecordInsn, /*DefineMI*/1, /*MI*/0, /*OpIdx*/4, // MIs[1]
        GIM_CheckOpcode, /*MI*/1, TargetOpcode::G_CONSTANT,
        // MIs[1] Operand 1
        // No operand predicates
        GIM_CheckIsSafeToFold, /*InsnID*/1,
        // (intrinsic_wo_chain:{ *:[v4i64] } 6369:{ *:[iPTR] }, VR256:{ *:[v4i64] }:$src1, VR256:{ *:[v4i64] }:$src2, (imm:{ *:[i8] }):$src3)  =>  (VPCLMULQDQYrr:{ *:[v4i64] } VR256:{ *:[v4i64] }:$src1, VR256:{ *:[v4i64] }:$src2, (imm:{ *:[i8] }):$src3)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::VPCLMULQDQYrr,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/2, // src1
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/3, // src2
        GIR_CopyConstantAsSImm, /*NewInsnID*/0, /*OldInsnID*/1, // src3
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 2635,
        GIR_Done,
      // Label 562: @16951
      GIM_Try, /*On fail goto*//*Label 563*/ 17019, // Rule ID 11186 //
        GIM_CheckFeatures, GIFBS_HasAVX512_HasVPCLMULQDQ,
        GIM_CheckIntrinsicID, /*MI*/0, /*Op*/1, Intrinsic::x86_pclmulqdq_512,
        GIM_CheckType, /*MI*/0, /*Op*/0, /*Type*/GILLT_v8s64,
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v8s64,
        GIM_CheckType, /*MI*/0, /*Op*/3, /*Type*/GILLT_v8s64,
        GIM_CheckType, /*MI*/0, /*Op*/4, /*Type*/GILLT_s8,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR512RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR512RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/3, /*RC*/X86::VR512RegClassID,
        GIM_RecordInsn, /*DefineMI*/1, /*MI*/0, /*OpIdx*/4, // MIs[1]
        GIM_CheckOpcode, /*MI*/1, TargetOpcode::G_CONSTANT,
        // MIs[1] Operand 1
        // No operand predicates
        GIM_CheckIsSafeToFold, /*InsnID*/1,
        // (intrinsic_wo_chain:{ *:[v8i64] } 6370:{ *:[iPTR] }, VR512:{ *:[v8i64] }:$src1, VR512:{ *:[v8i64] }:$src2, (imm:{ *:[i8] }):$src3)  =>  (VPCLMULQDQZrr:{ *:[v8i64] } VR512:{ *:[v8i64] }:$src1, VR512:{ *:[v8i64] }:$src2, (imm:{ *:[i8] }):$src3)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::VPCLMULQDQZrr,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/2, // src1
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/3, // src2
        GIR_CopyConstantAsSImm, /*NewInsnID*/0, /*OldInsnID*/1, // src3
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 11186,
        GIR_Done,
      // Label 563: @17019
      GIM_Try, /*On fail goto*//*Label 564*/ 17087, // Rule ID 11188 //
        GIM_CheckFeatures, GIFBS_HasVLX_HasVPCLMULQDQ,
        GIM_CheckIntrinsicID, /*MI*/0, /*Op*/1, Intrinsic::x86_pclmulqdq,
        GIM_CheckType, /*MI*/0, /*Op*/0, /*Type*/GILLT_v2s64,
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v2s64,
        GIM_CheckType, /*MI*/0, /*Op*/3, /*Type*/GILLT_v2s64,
        GIM_CheckType, /*MI*/0, /*Op*/4, /*Type*/GILLT_s8,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128XRegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR128XRegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/3, /*RC*/X86::VR128XRegClassID,
        GIM_RecordInsn, /*DefineMI*/1, /*MI*/0, /*OpIdx*/4, // MIs[1]
        GIM_CheckOpcode, /*MI*/1, TargetOpcode::G_CONSTANT,
        // MIs[1] Operand 1
        // No operand predicates
        GIM_CheckIsSafeToFold, /*InsnID*/1,
        // (intrinsic_wo_chain:{ *:[v2i64] } 6368:{ *:[iPTR] }, VR128X:{ *:[v2i64] }:$src1, VR128X:{ *:[v2i64] }:$src2, (imm:{ *:[i8] }):$src3)  =>  (VPCLMULQDQZ128rr:{ *:[v2i64] } VR128X:{ *:[v2i64] }:$src1, VR128X:{ *:[v2i64] }:$src2, (imm:{ *:[i8] }):$src3)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::VPCLMULQDQZ128rr,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/2, // src1
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/3, // src2
        GIR_CopyConstantAsSImm, /*NewInsnID*/0, /*OldInsnID*/1, // src3
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 11188,
        GIR_Done,
      // Label 564: @17087
      GIM_Try, /*On fail goto*//*Label 565*/ 17155, // Rule ID 11190 //
        GIM_CheckFeatures, GIFBS_HasVLX_HasVPCLMULQDQ,
        GIM_CheckIntrinsicID, /*MI*/0, /*Op*/1, Intrinsic::x86_pclmulqdq_256,
        GIM_CheckType, /*MI*/0, /*Op*/0, /*Type*/GILLT_v4s64,
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v4s64,
        GIM_CheckType, /*MI*/0, /*Op*/3, /*Type*/GILLT_v4s64,
        GIM_CheckType, /*MI*/0, /*Op*/4, /*Type*/GILLT_s8,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR256XRegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR256XRegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/3, /*RC*/X86::VR256XRegClassID,
        GIM_RecordInsn, /*DefineMI*/1, /*MI*/0, /*OpIdx*/4, // MIs[1]
        GIM_CheckOpcode, /*MI*/1, TargetOpcode::G_CONSTANT,
        // MIs[1] Operand 1
        // No operand predicates
        GIM_CheckIsSafeToFold, /*InsnID*/1,
        // (intrinsic_wo_chain:{ *:[v4i64] } 6369:{ *:[iPTR] }, VR256X:{ *:[v4i64] }:$src1, VR256X:{ *:[v4i64] }:$src2, (imm:{ *:[i8] }):$src3)  =>  (VPCLMULQDQZ256rr:{ *:[v4i64] } VR256X:{ *:[v4i64] }:$src1, VR256X:{ *:[v4i64] }:$src2, (imm:{ *:[i8] }):$src3)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::VPCLMULQDQZ256rr,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/2, // src1
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/3, // src2
        GIR_CopyConstantAsSImm, /*NewInsnID*/0, /*OldInsnID*/1, // src3
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 11190,
        GIR_Done,
      // Label 565: @17155
      GIM_Try, /*On fail goto*//*Label 566*/ 17219, // Rule ID 1215 //
        GIM_CheckFeatures, GIFBS_HasXOP,
        GIM_CheckIntrinsicID, /*MI*/0, /*Op*/1, Intrinsic::x86_xop_vpmadcswd,
        GIM_CheckType, /*MI*/0, /*Op*/0, /*Type*/GILLT_v4s32,
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v8s16,
        GIM_CheckType, /*MI*/0, /*Op*/3, /*Type*/GILLT_v8s16,
        GIM_CheckType, /*MI*/0, /*Op*/4, /*Type*/GILLT_v4s32,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/3, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/4, /*RC*/X86::VR128RegClassID,
        // (intrinsic_wo_chain:{ *:[v4i32] } 6657:{ *:[iPTR] }, VR128:{ *:[v8i16] }:$src1, VR128:{ *:[v8i16] }:$src2, VR128:{ *:[v4i32] }:$src3)  =>  (VPMADCSWDrr:{ *:[v4i32] } VR128:{ *:[v8i16] }:$src1, VR128:{ *:[v8i16] }:$src2, VR128:{ *:[v4i32] }:$src3)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::VPMADCSWDrr,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/2, // src1
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/3, // src2
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/4, // src3
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 1215,
        GIR_Done,
      // Label 566: @17219
      GIM_Try, /*On fail goto*//*Label 567*/ 17283, // Rule ID 1217 //
        GIM_CheckFeatures, GIFBS_HasXOP,
        GIM_CheckIntrinsicID, /*MI*/0, /*Op*/1, Intrinsic::x86_xop_vpmadcsswd,
        GIM_CheckType, /*MI*/0, /*Op*/0, /*Type*/GILLT_v4s32,
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v8s16,
        GIM_CheckType, /*MI*/0, /*Op*/3, /*Type*/GILLT_v8s16,
        GIM_CheckType, /*MI*/0, /*Op*/4, /*Type*/GILLT_v4s32,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/3, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/4, /*RC*/X86::VR128RegClassID,
        // (intrinsic_wo_chain:{ *:[v4i32] } 6656:{ *:[iPTR] }, VR128:{ *:[v8i16] }:$src1, VR128:{ *:[v8i16] }:$src2, VR128:{ *:[v4i32] }:$src3)  =>  (VPMADCSSWDrr:{ *:[v4i32] } VR128:{ *:[v8i16] }:$src1, VR128:{ *:[v8i16] }:$src2, VR128:{ *:[v4i32] }:$src3)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::VPMADCSSWDrr,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/2, // src1
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/3, // src2
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/4, // src3
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 1217,
        GIR_Done,
      // Label 567: @17283
      GIM_Try, /*On fail goto*//*Label 568*/ 17347, // Rule ID 1219 //
        GIM_CheckFeatures, GIFBS_HasXOP,
        GIM_CheckIntrinsicID, /*MI*/0, /*Op*/1, Intrinsic::x86_xop_vpmacsww,
        GIM_CheckType, /*MI*/0, /*Op*/0, /*Type*/GILLT_v8s16,
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v8s16,
        GIM_CheckType, /*MI*/0, /*Op*/3, /*Type*/GILLT_v8s16,
        GIM_CheckType, /*MI*/0, /*Op*/4, /*Type*/GILLT_v8s16,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/3, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/4, /*RC*/X86::VR128RegClassID,
        // (intrinsic_wo_chain:{ *:[v8i16] } 6655:{ *:[iPTR] }, VR128:{ *:[v8i16] }:$src1, VR128:{ *:[v8i16] }:$src2, VR128:{ *:[v8i16] }:$src3)  =>  (VPMACSWWrr:{ *:[v8i16] } VR128:{ *:[v8i16] }:$src1, VR128:{ *:[v8i16] }:$src2, VR128:{ *:[v8i16] }:$src3)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::VPMACSWWrr,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/2, // src1
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/3, // src2
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/4, // src3
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 1219,
        GIR_Done,
      // Label 568: @17347
      GIM_Try, /*On fail goto*//*Label 569*/ 17411, // Rule ID 1221 //
        GIM_CheckFeatures, GIFBS_HasXOP,
        GIM_CheckIntrinsicID, /*MI*/0, /*Op*/1, Intrinsic::x86_xop_vpmacswd,
        GIM_CheckType, /*MI*/0, /*Op*/0, /*Type*/GILLT_v4s32,
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v8s16,
        GIM_CheckType, /*MI*/0, /*Op*/3, /*Type*/GILLT_v8s16,
        GIM_CheckType, /*MI*/0, /*Op*/4, /*Type*/GILLT_v4s32,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/3, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/4, /*RC*/X86::VR128RegClassID,
        // (intrinsic_wo_chain:{ *:[v4i32] } 6654:{ *:[iPTR] }, VR128:{ *:[v8i16] }:$src1, VR128:{ *:[v8i16] }:$src2, VR128:{ *:[v4i32] }:$src3)  =>  (VPMACSWDrr:{ *:[v4i32] } VR128:{ *:[v8i16] }:$src1, VR128:{ *:[v8i16] }:$src2, VR128:{ *:[v4i32] }:$src3)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::VPMACSWDrr,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/2, // src1
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/3, // src2
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/4, // src3
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 1221,
        GIR_Done,
      // Label 569: @17411
      GIM_Try, /*On fail goto*//*Label 570*/ 17475, // Rule ID 1223 //
        GIM_CheckFeatures, GIFBS_HasXOP,
        GIM_CheckIntrinsicID, /*MI*/0, /*Op*/1, Intrinsic::x86_xop_vpmacssww,
        GIM_CheckType, /*MI*/0, /*Op*/0, /*Type*/GILLT_v8s16,
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v8s16,
        GIM_CheckType, /*MI*/0, /*Op*/3, /*Type*/GILLT_v8s16,
        GIM_CheckType, /*MI*/0, /*Op*/4, /*Type*/GILLT_v8s16,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/3, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/4, /*RC*/X86::VR128RegClassID,
        // (intrinsic_wo_chain:{ *:[v8i16] } 6653:{ *:[iPTR] }, VR128:{ *:[v8i16] }:$src1, VR128:{ *:[v8i16] }:$src2, VR128:{ *:[v8i16] }:$src3)  =>  (VPMACSSWWrr:{ *:[v8i16] } VR128:{ *:[v8i16] }:$src1, VR128:{ *:[v8i16] }:$src2, VR128:{ *:[v8i16] }:$src3)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::VPMACSSWWrr,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/2, // src1
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/3, // src2
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/4, // src3
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 1223,
        GIR_Done,
      // Label 570: @17475
      GIM_Try, /*On fail goto*//*Label 571*/ 17539, // Rule ID 1225 //
        GIM_CheckFeatures, GIFBS_HasXOP,
        GIM_CheckIntrinsicID, /*MI*/0, /*Op*/1, Intrinsic::x86_xop_vpmacsswd,
        GIM_CheckType, /*MI*/0, /*Op*/0, /*Type*/GILLT_v4s32,
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v8s16,
        GIM_CheckType, /*MI*/0, /*Op*/3, /*Type*/GILLT_v8s16,
        GIM_CheckType, /*MI*/0, /*Op*/4, /*Type*/GILLT_v4s32,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/3, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/4, /*RC*/X86::VR128RegClassID,
        // (intrinsic_wo_chain:{ *:[v4i32] } 6652:{ *:[iPTR] }, VR128:{ *:[v8i16] }:$src1, VR128:{ *:[v8i16] }:$src2, VR128:{ *:[v4i32] }:$src3)  =>  (VPMACSSWDrr:{ *:[v4i32] } VR128:{ *:[v8i16] }:$src1, VR128:{ *:[v8i16] }:$src2, VR128:{ *:[v4i32] }:$src3)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::VPMACSSWDrr,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/2, // src1
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/3, // src2
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/4, // src3
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 1225,
        GIR_Done,
      // Label 571: @17539
      GIM_Try, /*On fail goto*//*Label 572*/ 17603, // Rule ID 1227 //
        GIM_CheckFeatures, GIFBS_HasXOP,
        GIM_CheckIntrinsicID, /*MI*/0, /*Op*/1, Intrinsic::x86_xop_vpmacssdql,
        GIM_CheckType, /*MI*/0, /*Op*/0, /*Type*/GILLT_v2s64,
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v4s32,
        GIM_CheckType, /*MI*/0, /*Op*/3, /*Type*/GILLT_v4s32,
        GIM_CheckType, /*MI*/0, /*Op*/4, /*Type*/GILLT_v2s64,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/3, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/4, /*RC*/X86::VR128RegClassID,
        // (intrinsic_wo_chain:{ *:[v2i64] } 6651:{ *:[iPTR] }, VR128:{ *:[v4i32] }:$src1, VR128:{ *:[v4i32] }:$src2, VR128:{ *:[v2i64] }:$src3)  =>  (VPMACSSDQLrr:{ *:[v2i64] } VR128:{ *:[v4i32] }:$src1, VR128:{ *:[v4i32] }:$src2, VR128:{ *:[v2i64] }:$src3)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::VPMACSSDQLrr,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/2, // src1
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/3, // src2
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/4, // src3
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 1227,
        GIR_Done,
      // Label 572: @17603
      GIM_Try, /*On fail goto*//*Label 573*/ 17667, // Rule ID 1229 //
        GIM_CheckFeatures, GIFBS_HasXOP,
        GIM_CheckIntrinsicID, /*MI*/0, /*Op*/1, Intrinsic::x86_xop_vpmacssdqh,
        GIM_CheckType, /*MI*/0, /*Op*/0, /*Type*/GILLT_v2s64,
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v4s32,
        GIM_CheckType, /*MI*/0, /*Op*/3, /*Type*/GILLT_v4s32,
        GIM_CheckType, /*MI*/0, /*Op*/4, /*Type*/GILLT_v2s64,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/3, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/4, /*RC*/X86::VR128RegClassID,
        // (intrinsic_wo_chain:{ *:[v2i64] } 6650:{ *:[iPTR] }, VR128:{ *:[v4i32] }:$src1, VR128:{ *:[v4i32] }:$src2, VR128:{ *:[v2i64] }:$src3)  =>  (VPMACSSDQHrr:{ *:[v2i64] } VR128:{ *:[v4i32] }:$src1, VR128:{ *:[v4i32] }:$src2, VR128:{ *:[v2i64] }:$src3)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::VPMACSSDQHrr,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/2, // src1
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/3, // src2
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/4, // src3
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 1229,
        GIR_Done,
      // Label 573: @17667
      GIM_Try, /*On fail goto*//*Label 574*/ 17731, // Rule ID 1231 //
        GIM_CheckFeatures, GIFBS_HasXOP,
        GIM_CheckIntrinsicID, /*MI*/0, /*Op*/1, Intrinsic::x86_xop_vpmacssdd,
        GIM_CheckType, /*MI*/0, /*Op*/0, /*Type*/GILLT_v4s32,
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v4s32,
        GIM_CheckType, /*MI*/0, /*Op*/3, /*Type*/GILLT_v4s32,
        GIM_CheckType, /*MI*/0, /*Op*/4, /*Type*/GILLT_v4s32,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/3, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/4, /*RC*/X86::VR128RegClassID,
        // (intrinsic_wo_chain:{ *:[v4i32] } 6649:{ *:[iPTR] }, VR128:{ *:[v4i32] }:$src1, VR128:{ *:[v4i32] }:$src2, VR128:{ *:[v4i32] }:$src3)  =>  (VPMACSSDDrr:{ *:[v4i32] } VR128:{ *:[v4i32] }:$src1, VR128:{ *:[v4i32] }:$src2, VR128:{ *:[v4i32] }:$src3)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::VPMACSSDDrr,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/2, // src1
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/3, // src2
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/4, // src3
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 1231,
        GIR_Done,
      // Label 574: @17731
      GIM_Try, /*On fail goto*//*Label 575*/ 17795, // Rule ID 1233 //
        GIM_CheckFeatures, GIFBS_HasXOP,
        GIM_CheckIntrinsicID, /*MI*/0, /*Op*/1, Intrinsic::x86_xop_vpmacsdql,
        GIM_CheckType, /*MI*/0, /*Op*/0, /*Type*/GILLT_v2s64,
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v4s32,
        GIM_CheckType, /*MI*/0, /*Op*/3, /*Type*/GILLT_v4s32,
        GIM_CheckType, /*MI*/0, /*Op*/4, /*Type*/GILLT_v2s64,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/3, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/4, /*RC*/X86::VR128RegClassID,
        // (intrinsic_wo_chain:{ *:[v2i64] } 6648:{ *:[iPTR] }, VR128:{ *:[v4i32] }:$src1, VR128:{ *:[v4i32] }:$src2, VR128:{ *:[v2i64] }:$src3)  =>  (VPMACSDQLrr:{ *:[v2i64] } VR128:{ *:[v4i32] }:$src1, VR128:{ *:[v4i32] }:$src2, VR128:{ *:[v2i64] }:$src3)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::VPMACSDQLrr,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/2, // src1
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/3, // src2
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/4, // src3
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 1233,
        GIR_Done,
      // Label 575: @17795
      GIM_Try, /*On fail goto*//*Label 576*/ 17859, // Rule ID 1235 //
        GIM_CheckFeatures, GIFBS_HasXOP,
        GIM_CheckIntrinsicID, /*MI*/0, /*Op*/1, Intrinsic::x86_xop_vpmacsdqh,
        GIM_CheckType, /*MI*/0, /*Op*/0, /*Type*/GILLT_v2s64,
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v4s32,
        GIM_CheckType, /*MI*/0, /*Op*/3, /*Type*/GILLT_v4s32,
        GIM_CheckType, /*MI*/0, /*Op*/4, /*Type*/GILLT_v2s64,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/3, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/4, /*RC*/X86::VR128RegClassID,
        // (intrinsic_wo_chain:{ *:[v2i64] } 6647:{ *:[iPTR] }, VR128:{ *:[v4i32] }:$src1, VR128:{ *:[v4i32] }:$src2, VR128:{ *:[v2i64] }:$src3)  =>  (VPMACSDQHrr:{ *:[v2i64] } VR128:{ *:[v4i32] }:$src1, VR128:{ *:[v4i32] }:$src2, VR128:{ *:[v2i64] }:$src3)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::VPMACSDQHrr,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/2, // src1
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/3, // src2
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/4, // src3
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 1235,
        GIR_Done,
      // Label 576: @17859
      GIM_Try, /*On fail goto*//*Label 577*/ 17923, // Rule ID 1237 //
        GIM_CheckFeatures, GIFBS_HasXOP,
        GIM_CheckIntrinsicID, /*MI*/0, /*Op*/1, Intrinsic::x86_xop_vpmacsdd,
        GIM_CheckType, /*MI*/0, /*Op*/0, /*Type*/GILLT_v4s32,
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v4s32,
        GIM_CheckType, /*MI*/0, /*Op*/3, /*Type*/GILLT_v4s32,
        GIM_CheckType, /*MI*/0, /*Op*/4, /*Type*/GILLT_v4s32,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/3, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/4, /*RC*/X86::VR128RegClassID,
        // (intrinsic_wo_chain:{ *:[v4i32] } 6646:{ *:[iPTR] }, VR128:{ *:[v4i32] }:$src1, VR128:{ *:[v4i32] }:$src2, VR128:{ *:[v4i32] }:$src3)  =>  (VPMACSDDrr:{ *:[v4i32] } VR128:{ *:[v4i32] }:$src1, VR128:{ *:[v4i32] }:$src2, VR128:{ *:[v4i32] }:$src3)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::VPMACSDDrr,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/2, // src1
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/3, // src2
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/4, // src3
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 1237,
        GIR_Done,
      // Label 577: @17923
      GIM_Try, /*On fail goto*//*Label 578*/ 17987, // Rule ID 2553 //
        GIM_CheckFeatures, GIFBS_HasAVX,
        GIM_CheckIntrinsicID, /*MI*/0, /*Op*/1, Intrinsic::x86_sse41_blendvpd,
        GIM_CheckType, /*MI*/0, /*Op*/0, /*Type*/GILLT_v2s64,
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v2s64,
        GIM_CheckType, /*MI*/0, /*Op*/3, /*Type*/GILLT_v2s64,
        GIM_CheckType, /*MI*/0, /*Op*/4, /*Type*/GILLT_v2s64,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/3, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/4, /*RC*/X86::VR128RegClassID,
        // (intrinsic_wo_chain:{ *:[v2f64] } 6513:{ *:[iPTR] }, VR128:{ *:[v2f64] }:$src1, VR128:{ *:[v2f64] }:$src2, VR128:{ *:[v2f64] }:$src3)  =>  (VBLENDVPDrr:{ *:[v2f64] } VR128:{ *:[v2f64] }:$src1, VR128:{ *:[v2f64] }:$src2, VR128:{ *:[v2f64] }:$src3)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::VBLENDVPDrr,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/2, // src1
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/3, // src2
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/4, // src3
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 2553,
        GIR_Done,
      // Label 578: @17987
      GIM_Try, /*On fail goto*//*Label 579*/ 18051, // Rule ID 2555 //
        GIM_CheckFeatures, GIFBS_HasAVX,
        GIM_CheckIntrinsicID, /*MI*/0, /*Op*/1, Intrinsic::x86_avx_blendv_pd_256,
        GIM_CheckType, /*MI*/0, /*Op*/0, /*Type*/GILLT_v4s64,
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v4s64,
        GIM_CheckType, /*MI*/0, /*Op*/3, /*Type*/GILLT_v4s64,
        GIM_CheckType, /*MI*/0, /*Op*/4, /*Type*/GILLT_v4s64,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR256RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR256RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/3, /*RC*/X86::VR256RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/4, /*RC*/X86::VR256RegClassID,
        // (intrinsic_wo_chain:{ *:[v4f64] } 5479:{ *:[iPTR] }, VR256:{ *:[v4f64] }:$src1, VR256:{ *:[v4f64] }:$src2, VR256:{ *:[v4f64] }:$src3)  =>  (VBLENDVPDYrr:{ *:[v4f64] } VR256:{ *:[v4f64] }:$src1, VR256:{ *:[v4f64] }:$src2, VR256:{ *:[v4f64] }:$src3)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::VBLENDVPDYrr,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/2, // src1
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/3, // src2
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/4, // src3
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 2555,
        GIR_Done,
      // Label 579: @18051
      GIM_Try, /*On fail goto*//*Label 580*/ 18115, // Rule ID 2557 //
        GIM_CheckFeatures, GIFBS_HasAVX,
        GIM_CheckIntrinsicID, /*MI*/0, /*Op*/1, Intrinsic::x86_sse41_blendvps,
        GIM_CheckType, /*MI*/0, /*Op*/0, /*Type*/GILLT_v4s32,
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v4s32,
        GIM_CheckType, /*MI*/0, /*Op*/3, /*Type*/GILLT_v4s32,
        GIM_CheckType, /*MI*/0, /*Op*/4, /*Type*/GILLT_v4s32,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/3, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/4, /*RC*/X86::VR128RegClassID,
        // (intrinsic_wo_chain:{ *:[v4f32] } 6514:{ *:[iPTR] }, VR128:{ *:[v4f32] }:$src1, VR128:{ *:[v4f32] }:$src2, VR128:{ *:[v4f32] }:$src3)  =>  (VBLENDVPSrr:{ *:[v4f32] } VR128:{ *:[v4f32] }:$src1, VR128:{ *:[v4f32] }:$src2, VR128:{ *:[v4f32] }:$src3)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::VBLENDVPSrr,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/2, // src1
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/3, // src2
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/4, // src3
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 2557,
        GIR_Done,
      // Label 580: @18115
      GIM_Try, /*On fail goto*//*Label 581*/ 18179, // Rule ID 2559 //
        GIM_CheckFeatures, GIFBS_HasAVX,
        GIM_CheckIntrinsicID, /*MI*/0, /*Op*/1, Intrinsic::x86_avx_blendv_ps_256,
        GIM_CheckType, /*MI*/0, /*Op*/0, /*Type*/GILLT_v8s32,
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v8s32,
        GIM_CheckType, /*MI*/0, /*Op*/3, /*Type*/GILLT_v8s32,
        GIM_CheckType, /*MI*/0, /*Op*/4, /*Type*/GILLT_v8s32,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR256RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR256RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/3, /*RC*/X86::VR256RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/4, /*RC*/X86::VR256RegClassID,
        // (intrinsic_wo_chain:{ *:[v8f32] } 5480:{ *:[iPTR] }, VR256:{ *:[v8f32] }:$src1, VR256:{ *:[v8f32] }:$src2, VR256:{ *:[v8f32] }:$src3)  =>  (VBLENDVPSYrr:{ *:[v8f32] } VR256:{ *:[v8f32] }:$src1, VR256:{ *:[v8f32] }:$src2, VR256:{ *:[v8f32] }:$src3)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::VBLENDVPSYrr,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/2, // src1
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/3, // src2
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/4, // src3
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 2559,
        GIR_Done,
      // Label 581: @18179
      GIM_Try, /*On fail goto*//*Label 582*/ 18243, // Rule ID 2561 //
        GIM_CheckFeatures, GIFBS_HasAVX,
        GIM_CheckIntrinsicID, /*MI*/0, /*Op*/1, Intrinsic::x86_sse41_pblendvb,
        GIM_CheckType, /*MI*/0, /*Op*/0, /*Type*/GILLT_v16s8,
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v16s8,
        GIM_CheckType, /*MI*/0, /*Op*/3, /*Type*/GILLT_v16s8,
        GIM_CheckType, /*MI*/0, /*Op*/4, /*Type*/GILLT_v16s8,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/3, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/4, /*RC*/X86::VR128RegClassID,
        // (intrinsic_wo_chain:{ *:[v16i8] } 6520:{ *:[iPTR] }, VR128:{ *:[v16i8] }:$src1, VR128:{ *:[v16i8] }:$src2, VR128:{ *:[v16i8] }:$src3)  =>  (VPBLENDVBrr:{ *:[v16i8] } VR128:{ *:[v16i8] }:$src1, VR128:{ *:[v16i8] }:$src2, VR128:{ *:[v16i8] }:$src3)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::VPBLENDVBrr,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/2, // src1
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/3, // src2
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/4, // src3
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 2561,
        GIR_Done,
      // Label 582: @18243
      GIM_Try, /*On fail goto*//*Label 583*/ 18307, // Rule ID 2563 //
        GIM_CheckFeatures, GIFBS_HasAVX2,
        GIM_CheckIntrinsicID, /*MI*/0, /*Op*/1, Intrinsic::x86_avx2_pblendvb,
        GIM_CheckType, /*MI*/0, /*Op*/0, /*Type*/GILLT_v32s8,
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v32s8,
        GIM_CheckType, /*MI*/0, /*Op*/3, /*Type*/GILLT_v32s8,
        GIM_CheckType, /*MI*/0, /*Op*/4, /*Type*/GILLT_v32s8,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR256RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR256RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/3, /*RC*/X86::VR256RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/4, /*RC*/X86::VR256RegClassID,
        // (intrinsic_wo_chain:{ *:[v32i8] } 5564:{ *:[iPTR] }, VR256:{ *:[v32i8] }:$src1, VR256:{ *:[v32i8] }:$src2, VR256:{ *:[v32i8] }:$src3)  =>  (VPBLENDVBYrr:{ *:[v32i8] } VR256:{ *:[v32i8] }:$src1, VR256:{ *:[v32i8] }:$src2, VR256:{ *:[v32i8] }:$src3)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::VPBLENDVBYrr,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/2, // src1
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/3, // src2
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/4, // src3
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 2563,
        GIR_Done,
      // Label 583: @18307
      GIM_Reject,
    // Label 546: @18308
    GIM_Reject,
    // Label 8: @18309
    GIM_Try, /*On fail goto*//*Label 584*/ 18327, // Rule ID 1789 //
      GIM_CheckIntrinsicID, /*MI*/0, /*Op*/0, Intrinsic::x86_sse2_pause,
      // (intrinsic_void 6474:{ *:[iPTR] })  =>  (PAUSE)
      GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::PAUSE,
      GIR_MergeMemOperands, /*InsnID*/0, /*MergeInsnID's*/0, GIU_MergeMemOperands_EndOfList,
      GIR_EraseFromParent, /*InsnID*/0,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 1789,
      GIR_Done,
    // Label 584: @18327
    GIM_Try, /*On fail goto*//*Label 585*/ 18347, // Rule ID 1790 //
      GIM_CheckFeatures, GIFBS_HasSSE1,
      GIM_CheckIntrinsicID, /*MI*/0, /*Op*/0, Intrinsic::x86_sse_sfence,
      // (intrinsic_void 6434:{ *:[iPTR] })  =>  (SFENCE)
      GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::SFENCE,
      GIR_MergeMemOperands, /*InsnID*/0, /*MergeInsnID's*/0, GIU_MergeMemOperands_EndOfList,
      GIR_EraseFromParent, /*InsnID*/0,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 1790,
      GIR_Done,
    // Label 585: @18347
    GIM_Try, /*On fail goto*//*Label 586*/ 18367, // Rule ID 1791 //
      GIM_CheckFeatures, GIFBS_HasSSE2,
      GIM_CheckIntrinsicID, /*MI*/0, /*Op*/0, Intrinsic::x86_sse2_lfence,
      // (intrinsic_void 6461:{ *:[iPTR] })  =>  (LFENCE)
      GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::LFENCE,
      GIR_MergeMemOperands, /*InsnID*/0, /*MergeInsnID's*/0, GIU_MergeMemOperands_EndOfList,
      GIR_EraseFromParent, /*InsnID*/0,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 1791,
      GIR_Done,
    // Label 586: @18367
    GIM_Try, /*On fail goto*//*Label 587*/ 18387, // Rule ID 1792 //
      GIM_CheckFeatures, GIFBS_HasMFence,
      GIM_CheckIntrinsicID, /*MI*/0, /*Op*/0, Intrinsic::x86_sse2_mfence,
      // (intrinsic_void 6465:{ *:[iPTR] })  =>  (MFENCE)
      GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::MFENCE,
      GIR_MergeMemOperands, /*InsnID*/0, /*MergeInsnID's*/0, GIU_MergeMemOperands_EndOfList,
      GIR_EraseFromParent, /*InsnID*/0,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 1792,
      GIR_Done,
    // Label 587: @18387
    GIM_Try, /*On fail goto*//*Label 588*/ 18407, // Rule ID 2673 //
      GIM_CheckFeatures, GIFBS_HasAVX,
      GIM_CheckIntrinsicID, /*MI*/0, /*Op*/0, Intrinsic::x86_avx_vzeroall,
      // (intrinsic_void 5531:{ *:[iPTR] })  =>  (VZEROALL)
      GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::VZEROALL,
      GIR_MergeMemOperands, /*InsnID*/0, /*MergeInsnID's*/0, GIU_MergeMemOperands_EndOfList,
      GIR_EraseFromParent, /*InsnID*/0,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 2673,
      GIR_Done,
    // Label 588: @18407
    GIM_Try, /*On fail goto*//*Label 589*/ 18427, // Rule ID 2674 //
      GIM_CheckFeatures, GIFBS_HasAVX,
      GIM_CheckIntrinsicID, /*MI*/0, /*Op*/0, Intrinsic::x86_avx_vzeroupper,
      // (intrinsic_void 5532:{ *:[iPTR] })  =>  (VZEROUPPER)
      GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::VZEROUPPER,
      GIR_MergeMemOperands, /*InsnID*/0, /*MergeInsnID's*/0, GIU_MergeMemOperands_EndOfList,
      GIR_EraseFromParent, /*InsnID*/0,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 2674,
      GIR_Done,
    // Label 589: @18427
    GIM_Try, /*On fail goto*//*Label 590*/ 18447, // Rule ID 11762 //
      GIM_CheckFeatures, GIFBS_HasMMX,
      GIM_CheckIntrinsicID, /*MI*/0, /*Op*/0, Intrinsic::x86_mmx_emms,
      // (intrinsic_void 6294:{ *:[iPTR] })  =>  (MMX_EMMS)
      GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::MMX_EMMS,
      GIR_MergeMemOperands, /*InsnID*/0, /*MergeInsnID's*/0, GIU_MergeMemOperands_EndOfList,
      GIR_EraseFromParent, /*InsnID*/0,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 11762,
      GIR_Done,
    // Label 590: @18447
    GIM_Try, /*On fail goto*//*Label 591*/ 18467, // Rule ID 11981 //
      GIM_CheckFeatures, GIFBS_Has3DNow,
      GIM_CheckIntrinsicID, /*MI*/0, /*Op*/0, Intrinsic::x86_mmx_femms,
      // (intrinsic_void 6295:{ *:[iPTR] })  =>  (FEMMS)
      GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::FEMMS,
      GIR_MergeMemOperands, /*InsnID*/0, /*MergeInsnID's*/0, GIU_MergeMemOperands_EndOfList,
      GIR_EraseFromParent, /*InsnID*/0,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 11981,
      GIR_Done,
    // Label 591: @18467
    GIM_Try, /*On fail goto*//*Label 592*/ 18487, // Rule ID 11996 //
      GIM_CheckFeatures, GIFBS_HasRTM,
      GIM_CheckIntrinsicID, /*MI*/0, /*Op*/0, Intrinsic::x86_xend,
      // (intrinsic_void 6611:{ *:[iPTR] })  =>  (XEND)
      GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::XEND,
      GIR_MergeMemOperands, /*InsnID*/0, /*MergeInsnID's*/0, GIU_MergeMemOperands_EndOfList,
      GIR_EraseFromParent, /*InsnID*/0,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 11996,
      GIR_Done,
    // Label 592: @18487
    GIM_Try, /*On fail goto*//*Label 593*/ 18505, // Rule ID 12005 //
      GIM_CheckIntrinsicID, /*MI*/0, /*Op*/0, Intrinsic::x86_wbinvd,
      // (intrinsic_void 6598:{ *:[iPTR] })  =>  (WBINVD)
      GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::WBINVD,
      GIR_MergeMemOperands, /*InsnID*/0, /*MergeInsnID's*/0, GIU_MergeMemOperands_EndOfList,
      GIR_EraseFromParent, /*InsnID*/0,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 12005,
      GIR_Done,
    // Label 593: @18505
    GIM_Try, /*On fail goto*//*Label 594*/ 18525, // Rule ID 12006 //
      GIM_CheckFeatures, GIFBS_HasWBNOINVD,
      GIM_CheckIntrinsicID, /*MI*/0, /*Op*/0, Intrinsic::x86_wbnoinvd,
      // (intrinsic_void 6599:{ *:[iPTR] })  =>  (WBNOINVD)
      GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::WBNOINVD,
      GIR_MergeMemOperands, /*InsnID*/0, /*MergeInsnID's*/0, GIU_MergeMemOperands_EndOfList,
      GIR_EraseFromParent, /*InsnID*/0,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 12006,
      GIR_Done,
    // Label 594: @18525
    GIM_Try, /*On fail goto*//*Label 595*/ 18543, // Rule ID 12011 //
      GIM_CheckIntrinsicID, /*MI*/0, /*Op*/0, Intrinsic::x86_saveprevssp,
      // (intrinsic_void 6391:{ *:[iPTR] })  =>  (SAVEPREVSSP)
      GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::SAVEPREVSSP,
      GIR_MergeMemOperands, /*InsnID*/0, /*MergeInsnID's*/0, GIU_MergeMemOperands_EndOfList,
      GIR_EraseFromParent, /*InsnID*/0,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 12011,
      GIR_Done,
    // Label 595: @18543
    GIM_Try, /*On fail goto*//*Label 596*/ 18561, // Rule ID 12017 //
      GIM_CheckIntrinsicID, /*MI*/0, /*Op*/0, Intrinsic::x86_setssbsy,
      // (intrinsic_void 6396:{ *:[iPTR] })  =>  (SETSSBSY)
      GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::SETSSBSY,
      GIR_MergeMemOperands, /*InsnID*/0, /*MergeInsnID's*/0, GIU_MergeMemOperands_EndOfList,
      GIR_EraseFromParent, /*InsnID*/0,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 12017,
      GIR_Done,
    // Label 596: @18561
    GIM_Try, /*On fail goto*//*Label 597*/ 19521,
      GIM_CheckNumOperands, /*MI*/0, /*Expected*/2,
      GIM_Try, /*On fail goto*//*Label 598*/ 18592, // Rule ID 12002 //
        GIM_CheckIntrinsicID, /*MI*/0, /*Op*/0, Intrinsic::x86_int,
        GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_s8,
        GIM_CheckConstantInt, /*MI*/0, /*Op*/1, 3,
        // (intrinsic_void 6287:{ *:[iPTR] }, 3:{ *:[i8] })  =>  (INT3)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::INT3,
        GIR_MergeMemOperands, /*InsnID*/0, /*MergeInsnID's*/0, GIU_MergeMemOperands_EndOfList,
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 12002,
        GIR_Done,
      // Label 598: @18592
      GIM_Try, /*On fail goto*//*Label 599*/ 18629, // Rule ID 11998 //
        GIM_CheckFeatures, GIFBS_HasRTM,
        GIM_CheckIntrinsicID, /*MI*/0, /*Op*/0, Intrinsic::x86_xabort,
        GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_s8,
        GIM_RecordInsn, /*DefineMI*/1, /*MI*/0, /*OpIdx*/1, // MIs[1]
        GIM_CheckOpcode, /*MI*/1, TargetOpcode::G_CONSTANT,
        // MIs[1] Operand 1
        // No operand predicates
        GIM_CheckIsSafeToFold, /*InsnID*/1,
        // (intrinsic_void 6609:{ *:[iPTR] }, (imm:{ *:[i8] }):$imm)  =>  (XABORT (imm:{ *:[i8] }):$imm)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::XABORT,
        GIR_CopyConstantAsSImm, /*NewInsnID*/0, /*OldInsnID*/1, // imm
        GIR_MergeMemOperands, /*InsnID*/0, /*MergeInsnID's*/0, 1, GIU_MergeMemOperands_EndOfList,
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 11998,
        GIR_Done,
      // Label 599: @18629
      GIM_Try, /*On fail goto*//*Label 600*/ 18664, // Rule ID 12003 //
        GIM_CheckIntrinsicID, /*MI*/0, /*Op*/0, Intrinsic::x86_int,
        GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_s8,
        GIM_RecordInsn, /*DefineMI*/1, /*MI*/0, /*OpIdx*/1, // MIs[1]
        GIM_CheckOpcode, /*MI*/1, TargetOpcode::G_CONSTANT,
        // MIs[1] Operand 1
        // No operand predicates
        GIM_CheckIsSafeToFold, /*InsnID*/1,
        // (intrinsic_void 6287:{ *:[iPTR] }, (imm:{ *:[i8] }):$trap)  =>  (INT (imm:{ *:[i8] }):$trap)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::INT,
        GIR_CopyConstantAsSImm, /*NewInsnID*/0, /*OldInsnID*/1, // trap
        GIR_MergeMemOperands, /*InsnID*/0, /*MergeInsnID's*/0, 1, GIU_MergeMemOperands_EndOfList,
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 12003,
        GIR_Done,
      // Label 600: @18664
      GIM_Try, /*On fail goto*//*Label 601*/ 18696, // Rule ID 1 //
        GIM_CheckFeatures, GIFBS_Not64BitMode,
        GIM_CheckIntrinsicID, /*MI*/0, /*Op*/1, Intrinsic::x86_flags_read_u32,
        GIM_CheckType, /*MI*/0, /*Op*/0, /*Type*/GILLT_s32,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::GR32RegClassID,
        // (intrinsic_w_chain:{ *:[i32] } 6277:{ *:[iPTR] })  =>  (RDFLAGS32:{ *:[i32] }:{ *:[i32] })
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::RDFLAGS32,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_MergeMemOperands, /*InsnID*/0, /*MergeInsnID's*/0, GIU_MergeMemOperands_EndOfList,
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 1,
        GIR_Done,
      // Label 601: @18696
      GIM_Try, /*On fail goto*//*Label 602*/ 18728, // Rule ID 2 //
        GIM_CheckFeatures, GIFBS_In64BitMode,
        GIM_CheckIntrinsicID, /*MI*/0, /*Op*/1, Intrinsic::x86_flags_read_u64,
        GIM_CheckType, /*MI*/0, /*Op*/0, /*Type*/GILLT_s64,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::GR64RegClassID,
        // (intrinsic_w_chain:{ *:[i64] } 6278:{ *:[iPTR] })  =>  (RDFLAGS64:{ *:[i64] }:{ *:[i32] })
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::RDFLAGS64,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_MergeMemOperands, /*InsnID*/0, /*MergeInsnID's*/0, GIU_MergeMemOperands_EndOfList,
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 2,
        GIR_Done,
      // Label 602: @18728
      GIM_Try, /*On fail goto*//*Label 603*/ 18760, // Rule ID 95 //
        GIM_CheckFeatures, GIFBS_HasLWP,
        GIM_CheckIntrinsicID, /*MI*/0, /*Op*/1, Intrinsic::x86_slwpcb,
        GIM_CheckType, /*MI*/0, /*Op*/0, /*Type*/GILLT_s32,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::GR32RegClassID,
        // (intrinsic_w_chain:{ *:[i32] } 6404:{ *:[iPTR] })  =>  (SLWPCB:{ *:[i32] })
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::SLWPCB,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_MergeMemOperands, /*InsnID*/0, /*MergeInsnID's*/0, GIU_MergeMemOperands_EndOfList,
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 95,
        GIR_Done,
      // Label 603: @18760
      GIM_Try, /*On fail goto*//*Label 604*/ 18792, // Rule ID 97 //
        GIM_CheckFeatures, GIFBS_HasLWP,
        GIM_CheckIntrinsicID, /*MI*/0, /*Op*/1, Intrinsic::x86_slwpcb,
        GIM_CheckType, /*MI*/0, /*Op*/0, /*Type*/GILLT_s64,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::GR64RegClassID,
        // (intrinsic_w_chain:{ *:[i64] } 6404:{ *:[iPTR] })  =>  (SLWPCB64:{ *:[i64] })
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::SLWPCB64,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_MergeMemOperands, /*InsnID*/0, /*MergeInsnID's*/0, GIU_MergeMemOperands_EndOfList,
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 97,
        GIR_Done,
      // Label 604: @18792
      GIM_Try, /*On fail goto*//*Label 605*/ 18824, // Rule ID 11995 //
        GIM_CheckFeatures, GIFBS_HasRTM,
        GIM_CheckIntrinsicID, /*MI*/0, /*Op*/1, Intrinsic::x86_xbegin,
        GIM_CheckType, /*MI*/0, /*Op*/0, /*Type*/GILLT_s32,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::GR32RegClassID,
        // (intrinsic_w_chain:{ *:[i32] } 6610:{ *:[iPTR] })  =>  (XBEGIN:{ *:[i32] })
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::XBEGIN,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_MergeMemOperands, /*InsnID*/0, /*MergeInsnID's*/0, GIU_MergeMemOperands_EndOfList,
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 11995,
        GIR_Done,
      // Label 605: @18824
      GIM_Try, /*On fail goto*//*Label 606*/ 18854, // Rule ID 12033 //
        GIM_CheckIntrinsicID, /*MI*/0, /*Op*/1, Intrinsic::x86_rdpkru,
        GIM_CheckType, /*MI*/0, /*Op*/0, /*Type*/GILLT_s32,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::GR32RegClassID,
        // (intrinsic_w_chain:{ *:[i32] } 6378:{ *:[iPTR] })  =>  (RDPKRU:{ *:[i32] })
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::RDPKRU,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_MergeMemOperands, /*InsnID*/0, /*MergeInsnID's*/0, GIU_MergeMemOperands_EndOfList,
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 12033,
        GIR_Done,
      // Label 606: @18854
      GIM_Try, /*On fail goto*//*Label 607*/ 18886, // Rule ID 12034 //
        GIM_CheckFeatures, GIFBS_HasFSGSBase_In64BitMode,
        GIM_CheckIntrinsicID, /*MI*/0, /*Op*/1, Intrinsic::x86_rdfsbase_32,
        GIM_CheckType, /*MI*/0, /*Op*/0, /*Type*/GILLT_s32,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::GR32RegClassID,
        // (intrinsic_w_chain:{ *:[i32] } 6373:{ *:[iPTR] })  =>  (RDFSBASE:{ *:[i32] })
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::RDFSBASE,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_MergeMemOperands, /*InsnID*/0, /*MergeInsnID's*/0, GIU_MergeMemOperands_EndOfList,
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 12034,
        GIR_Done,
      // Label 607: @18886
      GIM_Try, /*On fail goto*//*Label 608*/ 18918, // Rule ID 12035 //
        GIM_CheckFeatures, GIFBS_HasFSGSBase_In64BitMode,
        GIM_CheckIntrinsicID, /*MI*/0, /*Op*/1, Intrinsic::x86_rdfsbase_64,
        GIM_CheckType, /*MI*/0, /*Op*/0, /*Type*/GILLT_s64,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::GR64RegClassID,
        // (intrinsic_w_chain:{ *:[i64] } 6374:{ *:[iPTR] })  =>  (RDFSBASE64:{ *:[i64] })
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::RDFSBASE64,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_MergeMemOperands, /*InsnID*/0, /*MergeInsnID's*/0, GIU_MergeMemOperands_EndOfList,
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 12035,
        GIR_Done,
      // Label 608: @18918
      GIM_Try, /*On fail goto*//*Label 609*/ 18950, // Rule ID 12036 //
        GIM_CheckFeatures, GIFBS_HasFSGSBase_In64BitMode,
        GIM_CheckIntrinsicID, /*MI*/0, /*Op*/1, Intrinsic::x86_rdgsbase_32,
        GIM_CheckType, /*MI*/0, /*Op*/0, /*Type*/GILLT_s32,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::GR32RegClassID,
        // (intrinsic_w_chain:{ *:[i32] } 6375:{ *:[iPTR] })  =>  (RDGSBASE:{ *:[i32] })
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::RDGSBASE,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_MergeMemOperands, /*InsnID*/0, /*MergeInsnID's*/0, GIU_MergeMemOperands_EndOfList,
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 12036,
        GIR_Done,
      // Label 609: @18950
      GIM_Try, /*On fail goto*//*Label 610*/ 18982, // Rule ID 12037 //
        GIM_CheckFeatures, GIFBS_HasFSGSBase_In64BitMode,
        GIM_CheckIntrinsicID, /*MI*/0, /*Op*/1, Intrinsic::x86_rdgsbase_64,
        GIM_CheckType, /*MI*/0, /*Op*/0, /*Type*/GILLT_s64,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::GR64RegClassID,
        // (intrinsic_w_chain:{ *:[i64] } 6376:{ *:[iPTR] })  =>  (RDGSBASE64:{ *:[i64] })
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::RDGSBASE64,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_MergeMemOperands, /*InsnID*/0, /*MergeInsnID's*/0, GIU_MergeMemOperands_EndOfList,
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 12037,
        GIR_Done,
      // Label 610: @18982
      GIM_Try, /*On fail goto*//*Label 611*/ 19014, // Rule ID 12043 //
        GIM_CheckFeatures, GIFBS_HasRDPID_Not64BitMode,
        GIM_CheckIntrinsicID, /*MI*/0, /*Op*/1, Intrinsic::x86_rdpid,
        GIM_CheckType, /*MI*/0, /*Op*/0, /*Type*/GILLT_s32,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::GR32RegClassID,
        // (intrinsic_w_chain:{ *:[i32] } 6377:{ *:[iPTR] })  =>  (RDPID32:{ *:[i32] })
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::RDPID32,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_MergeMemOperands, /*InsnID*/0, /*MergeInsnID's*/0, GIU_MergeMemOperands_EndOfList,
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 12043,
        GIR_Done,
      // Label 611: @19014
      GIM_Try, /*On fail goto*//*Label 612*/ 19046, // Rule ID 3 //
        GIM_CheckFeatures, GIFBS_Not64BitMode,
        GIM_CheckIntrinsicID, /*MI*/0, /*Op*/0, Intrinsic::x86_flags_write_u32,
        GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_s32,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::GR32RegClassID,
        // (intrinsic_void 6279:{ *:[iPTR] }, GR32:{ *:[i32] }:$src)  =>  (WRFLAGS32:{ *:[i32] } GR32:{ *:[i32] }:$src)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::WRFLAGS32,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // src
        GIR_MergeMemOperands, /*InsnID*/0, /*MergeInsnID's*/0, GIU_MergeMemOperands_EndOfList,
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 3,
        GIR_Done,
      // Label 612: @19046
      GIM_Try, /*On fail goto*//*Label 613*/ 19078, // Rule ID 4 //
        GIM_CheckFeatures, GIFBS_In64BitMode,
        GIM_CheckIntrinsicID, /*MI*/0, /*Op*/0, Intrinsic::x86_flags_write_u64,
        GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_s64,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::GR64RegClassID,
        // (intrinsic_void 6280:{ *:[iPTR] }, GR64:{ *:[i64] }:$src)  =>  (WRFLAGS64:{ *:[i64] } GR64:{ *:[i64] }:$src)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::WRFLAGS64,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // src
        GIR_MergeMemOperands, /*InsnID*/0, /*MergeInsnID's*/0, GIU_MergeMemOperands_EndOfList,
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 4,
        GIR_Done,
      // Label 613: @19078
      GIM_Try, /*On fail goto*//*Label 614*/ 19110, // Rule ID 94 //
        GIM_CheckFeatures, GIFBS_HasLWP,
        GIM_CheckIntrinsicID, /*MI*/0, /*Op*/0, Intrinsic::x86_llwpcb,
        GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_s32,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::GR32RegClassID,
        // (intrinsic_void 6289:{ *:[iPTR] }, GR32:{ *:[i32] }:$src)  =>  (LLWPCB GR32:{ *:[i32] }:$src)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::LLWPCB,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // src
        GIR_MergeMemOperands, /*InsnID*/0, /*MergeInsnID's*/0, GIU_MergeMemOperands_EndOfList,
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 94,
        GIR_Done,
      // Label 614: @19110
      GIM_Try, /*On fail goto*//*Label 615*/ 19142, // Rule ID 96 //
        GIM_CheckFeatures, GIFBS_HasLWP,
        GIM_CheckIntrinsicID, /*MI*/0, /*Op*/0, Intrinsic::x86_llwpcb,
        GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_s64,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::GR64RegClassID,
        // (intrinsic_void 6289:{ *:[iPTR] }, GR64:{ *:[i64] }:$src)  =>  (LLWPCB64 GR64:{ *:[i64] }:$src)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::LLWPCB64,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // src
        GIR_MergeMemOperands, /*InsnID*/0, /*MergeInsnID's*/0, GIU_MergeMemOperands_EndOfList,
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 96,
        GIR_Done,
      // Label 615: @19142
      GIM_Try, /*On fail goto*//*Label 616*/ 19174, // Rule ID 108 //
        GIM_CheckFeatures, GIFBS_HasWAITPKG_Not64BitMode,
        GIM_CheckIntrinsicID, /*MI*/0, /*Op*/0, Intrinsic::x86_umonitor,
        GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_s16,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::GR16RegClassID,
        // (intrinsic_void 6583:{ *:[iPTR] }, GR16:{ *:[i16] }:$src)  =>  (UMONITOR16 GR16:{ *:[i16] }:$src)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::UMONITOR16,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // src
        GIR_MergeMemOperands, /*InsnID*/0, /*MergeInsnID's*/0, GIU_MergeMemOperands_EndOfList,
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 108,
        GIR_Done,
      // Label 616: @19174
      GIM_Try, /*On fail goto*//*Label 617*/ 19206, // Rule ID 109 //
        GIM_CheckFeatures, GIFBS_HasWAITPKG,
        GIM_CheckIntrinsicID, /*MI*/0, /*Op*/0, Intrinsic::x86_umonitor,
        GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_s32,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::GR32RegClassID,
        // (intrinsic_void 6583:{ *:[iPTR] }, GR32:{ *:[i32] }:$src)  =>  (UMONITOR32 GR32:{ *:[i32] }:$src)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::UMONITOR32,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // src
        GIR_MergeMemOperands, /*InsnID*/0, /*MergeInsnID's*/0, GIU_MergeMemOperands_EndOfList,
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 109,
        GIR_Done,
      // Label 617: @19206
      GIM_Try, /*On fail goto*//*Label 618*/ 19238, // Rule ID 110 //
        GIM_CheckFeatures, GIFBS_HasWAITPKG_In64BitMode,
        GIM_CheckIntrinsicID, /*MI*/0, /*Op*/0, Intrinsic::x86_umonitor,
        GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_s64,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::GR64RegClassID,
        // (intrinsic_void 6583:{ *:[iPTR] }, GR64:{ *:[i64] }:$src)  =>  (UMONITOR64 GR64:{ *:[i64] }:$src)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::UMONITOR64,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // src
        GIR_MergeMemOperands, /*InsnID*/0, /*MergeInsnID's*/0, GIU_MergeMemOperands_EndOfList,
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 110,
        GIR_Done,
      // Label 618: @19238
      GIM_Try, /*On fail goto*//*Label 619*/ 19268, // Rule ID 12007 //
        GIM_CheckIntrinsicID, /*MI*/0, /*Op*/0, Intrinsic::x86_incsspd,
        GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_s32,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::GR32RegClassID,
        // (intrinsic_void 6285:{ *:[iPTR] }, GR32:{ *:[i32] }:$src)  =>  (INCSSPD GR32:{ *:[i32] }:$src)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::INCSSPD,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // src
        GIR_MergeMemOperands, /*InsnID*/0, /*MergeInsnID's*/0, GIU_MergeMemOperands_EndOfList,
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 12007,
        GIR_Done,
      // Label 619: @19268
      GIM_Try, /*On fail goto*//*Label 620*/ 19298, // Rule ID 12008 //
        GIM_CheckIntrinsicID, /*MI*/0, /*Op*/0, Intrinsic::x86_incsspq,
        GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_s64,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::GR64RegClassID,
        // (intrinsic_void 6286:{ *:[iPTR] }, GR64:{ *:[i64] }:$src)  =>  (INCSSPQ GR64:{ *:[i64] }:$src)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::INCSSPQ,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // src
        GIR_MergeMemOperands, /*InsnID*/0, /*MergeInsnID's*/0, GIU_MergeMemOperands_EndOfList,
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 12008,
        GIR_Done,
      // Label 620: @19298
      GIM_Try, /*On fail goto*//*Label 621*/ 19328, // Rule ID 12032 //
        GIM_CheckIntrinsicID, /*MI*/0, /*Op*/0, Intrinsic::x86_wrpkru,
        GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_s32,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::GR32RegClassID,
        // (intrinsic_void 6604:{ *:[iPTR] }, GR32:{ *:[i32] }:$src)  =>  (WRPKRU GR32:{ *:[i32] }:$src)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::WRPKRU,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // src
        GIR_MergeMemOperands, /*InsnID*/0, /*MergeInsnID's*/0, GIU_MergeMemOperands_EndOfList,
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 12032,
        GIR_Done,
      // Label 621: @19328
      GIM_Try, /*On fail goto*//*Label 622*/ 19360, // Rule ID 12038 //
        GIM_CheckFeatures, GIFBS_HasFSGSBase_In64BitMode,
        GIM_CheckIntrinsicID, /*MI*/0, /*Op*/0, Intrinsic::x86_wrfsbase_32,
        GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_s32,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::GR32RegClassID,
        // (intrinsic_void 6600:{ *:[iPTR] }, GR32:{ *:[i32] }:$src)  =>  (WRFSBASE GR32:{ *:[i32] }:$src)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::WRFSBASE,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // src
        GIR_MergeMemOperands, /*InsnID*/0, /*MergeInsnID's*/0, GIU_MergeMemOperands_EndOfList,
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 12038,
        GIR_Done,
      // Label 622: @19360
      GIM_Try, /*On fail goto*//*Label 623*/ 19392, // Rule ID 12039 //
        GIM_CheckFeatures, GIFBS_HasFSGSBase_In64BitMode,
        GIM_CheckIntrinsicID, /*MI*/0, /*Op*/0, Intrinsic::x86_wrfsbase_64,
        GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_s64,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::GR64RegClassID,
        // (intrinsic_void 6601:{ *:[iPTR] }, GR64:{ *:[i64] }:$src)  =>  (WRFSBASE64 GR64:{ *:[i64] }:$src)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::WRFSBASE64,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // src
        GIR_MergeMemOperands, /*InsnID*/0, /*MergeInsnID's*/0, GIU_MergeMemOperands_EndOfList,
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 12039,
        GIR_Done,
      // Label 623: @19392
      GIM_Try, /*On fail goto*//*Label 624*/ 19424, // Rule ID 12040 //
        GIM_CheckFeatures, GIFBS_HasFSGSBase_In64BitMode,
        GIM_CheckIntrinsicID, /*MI*/0, /*Op*/0, Intrinsic::x86_wrgsbase_32,
        GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_s32,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::GR32RegClassID,
        // (intrinsic_void 6602:{ *:[iPTR] }, GR32:{ *:[i32] }:$src)  =>  (WRGSBASE GR32:{ *:[i32] }:$src)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::WRGSBASE,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // src
        GIR_MergeMemOperands, /*InsnID*/0, /*MergeInsnID's*/0, GIU_MergeMemOperands_EndOfList,
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 12040,
        GIR_Done,
      // Label 624: @19424
      GIM_Try, /*On fail goto*//*Label 625*/ 19456, // Rule ID 12041 //
        GIM_CheckFeatures, GIFBS_HasFSGSBase_In64BitMode,
        GIM_CheckIntrinsicID, /*MI*/0, /*Op*/0, Intrinsic::x86_wrgsbase_64,
        GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_s64,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::GR64RegClassID,
        // (intrinsic_void 6603:{ *:[iPTR] }, GR64:{ *:[i64] }:$src)  =>  (WRGSBASE64 GR64:{ *:[i64] }:$src)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::WRGSBASE64,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // src
        GIR_MergeMemOperands, /*InsnID*/0, /*MergeInsnID's*/0, GIU_MergeMemOperands_EndOfList,
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 12041,
        GIR_Done,
      // Label 625: @19456
      GIM_Try, /*On fail goto*//*Label 626*/ 19488, // Rule ID 12046 //
        GIM_CheckFeatures, GIFBS_HasPTWRITE,
        GIM_CheckIntrinsicID, /*MI*/0, /*Op*/0, Intrinsic::x86_ptwrite32,
        GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_s32,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::GR32RegClassID,
        // (intrinsic_void 6371:{ *:[iPTR] }, GR32:{ *:[i32] }:$dst)  =>  (PTWRITEr GR32:{ *:[i32] }:$dst)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::PTWRITEr,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // dst
        GIR_MergeMemOperands, /*InsnID*/0, /*MergeInsnID's*/0, GIU_MergeMemOperands_EndOfList,
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 12046,
        GIR_Done,
      // Label 626: @19488
      GIM_Try, /*On fail goto*//*Label 627*/ 19520, // Rule ID 12047 //
        GIM_CheckFeatures, GIFBS_HasPTWRITE_In64BitMode,
        GIM_CheckIntrinsicID, /*MI*/0, /*Op*/0, Intrinsic::x86_ptwrite64,
        GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_s64,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::GR64RegClassID,
        // (intrinsic_void 6372:{ *:[iPTR] }, GR64:{ *:[i64] }:$dst)  =>  (PTWRITE64r GR64:{ *:[i64] }:$dst)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::PTWRITE64r,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // dst
        GIR_MergeMemOperands, /*InsnID*/0, /*MergeInsnID's*/0, GIU_MergeMemOperands_EndOfList,
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 12047,
        GIR_Done,
      // Label 627: @19520
      GIM_Reject,
    // Label 597: @19521
    GIM_Try, /*On fail goto*//*Label 628*/ 19611,
      GIM_CheckNumOperands, /*MI*/0, /*Expected*/3,
      GIM_Try, /*On fail goto*//*Label 629*/ 19568, // Rule ID 12009 //
        GIM_CheckIntrinsicID, /*MI*/0, /*Op*/1, Intrinsic::x86_rdsspd,
        GIM_CheckType, /*MI*/0, /*Op*/0, /*Type*/GILLT_s32,
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_s32,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::GR32RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::GR32RegClassID,
        // (intrinsic_w_chain:{ *:[i32] } 6386:{ *:[iPTR] }, GR32:{ *:[i32] }:$src)  =>  (RDSSPD:{ *:[i32] } GR32:{ *:[i32] }:$src)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::RDSSPD,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/2, // src
        GIR_MergeMemOperands, /*InsnID*/0, /*MergeInsnID's*/0, GIU_MergeMemOperands_EndOfList,
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 12009,
        GIR_Done,
      // Label 629: @19568
      GIM_Try, /*On fail goto*//*Label 630*/ 19610, // Rule ID 12010 //
        GIM_CheckIntrinsicID, /*MI*/0, /*Op*/1, Intrinsic::x86_rdsspq,
        GIM_CheckType, /*MI*/0, /*Op*/0, /*Type*/GILLT_s64,
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_s64,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::GR64RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::GR64RegClassID,
        // (intrinsic_w_chain:{ *:[i64] } 6387:{ *:[iPTR] }, GR64:{ *:[i64] }:$src)  =>  (RDSSPQ:{ *:[i64] } GR64:{ *:[i64] }:$src)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::RDSSPQ,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/2, // src
        GIR_MergeMemOperands, /*InsnID*/0, /*MergeInsnID's*/0, GIU_MergeMemOperands_EndOfList,
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 12010,
        GIR_Done,
      // Label 630: @19610
      GIM_Reject,
    // Label 628: @19611
    GIM_Try, /*On fail goto*//*Label 631*/ 19739,
      GIM_CheckNumOperands, /*MI*/0, /*Expected*/4,
      GIM_Try, /*On fail goto*//*Label 632*/ 19677, // Rule ID 102 //
        GIM_CheckFeatures, GIFBS_HasLWP,
        GIM_CheckIntrinsicID, /*MI*/0, /*Op*/0, Intrinsic::x86_lwpval32,
        GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_s32,
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_s32,
        GIM_CheckType, /*MI*/0, /*Op*/3, /*Type*/GILLT_s32,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::GR32RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::GR32RegClassID,
        GIM_RecordInsn, /*DefineMI*/1, /*MI*/0, /*OpIdx*/3, // MIs[1]
        GIM_CheckOpcode, /*MI*/1, TargetOpcode::G_CONSTANT,
        // MIs[1] Operand 1
        // No operand predicates
        GIM_CheckIsSafeToFold, /*InsnID*/1,
        // (intrinsic_void 6292:{ *:[iPTR] }, GR32:{ *:[i32] }:$src0, GR32:{ *:[i32] }:$src1, (imm:{ *:[i32] }):$cntl)  =>  (LWPVAL32rri GR32:{ *:[i32] }:$src0, GR32:{ *:[i32] }:$src1, (imm:{ *:[i32] }):$cntl)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::LWPVAL32rri,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // src0
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/2, // src1
        GIR_CopyConstantAsSImm, /*NewInsnID*/0, /*OldInsnID*/1, // cntl
        GIR_MergeMemOperands, /*InsnID*/0, /*MergeInsnID's*/0, 1, GIU_MergeMemOperands_EndOfList,
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 102,
        GIR_Done,
      // Label 632: @19677
      GIM_Try, /*On fail goto*//*Label 633*/ 19738, // Rule ID 104 //
        GIM_CheckFeatures, GIFBS_HasLWP,
        GIM_CheckIntrinsicID, /*MI*/0, /*Op*/0, Intrinsic::x86_lwpval64,
        GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_s64,
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_s32,
        GIM_CheckType, /*MI*/0, /*Op*/3, /*Type*/GILLT_s32,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::GR64RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::GR32RegClassID,
        GIM_RecordInsn, /*DefineMI*/1, /*MI*/0, /*OpIdx*/3, // MIs[1]
        GIM_CheckOpcode, /*MI*/1, TargetOpcode::G_CONSTANT,
        // MIs[1] Operand 1
        // No operand predicates
        GIM_CheckIsSafeToFold, /*InsnID*/1,
        // (intrinsic_void 6293:{ *:[iPTR] }, GR64:{ *:[i64] }:$src0, GR32:{ *:[i32] }:$src1, (imm:{ *:[i32] }):$cntl)  =>  (LWPVAL64rri GR64:{ *:[i64] }:$src0, GR32:{ *:[i32] }:$src1, (imm:{ *:[i32] }):$cntl)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::LWPVAL64rri,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // src0
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/2, // src1
        GIR_CopyConstantAsSImm, /*NewInsnID*/0, /*OldInsnID*/1, // cntl
        GIR_MergeMemOperands, /*InsnID*/0, /*MergeInsnID's*/0, 1, GIU_MergeMemOperands_EndOfList,
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 104,
        GIR_Done,
      // Label 633: @19738
      GIM_Reject,
    // Label 631: @19739
    GIM_Reject,
    // Label 9: @19740
    GIM_Try, /*On fail goto*//*Label 634*/ 19855,
      GIM_CheckType, /*MI*/0, /*Op*/0, /*Type*/GILLT_s32,
      GIM_SwitchType, /*MI*/0, /*Op*/1, /*[*/1, 3, /*)*//*default:*//*Label 637*/ 19854,
      /*GILLT_s8*//*Label 635*/ 19754,
      /*GILLT_s16*//*Label 636*/ 19812,
      // Label 635: @19754
      GIM_Try, /*On fail goto*//*Label 638*/ 19811,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::GR32RegClassID,
        GIM_Try, /*On fail goto*//*Label 639*/ 19797, // Rule ID 13702 //
          GIM_RecordInsn, /*DefineMI*/1, /*MI*/0, /*OpIdx*/1, // MIs[1]
          GIM_CheckOpcode, /*MI*/1, TargetOpcode::G_BITCAST,
          GIM_CheckType, /*MI*/1, /*Op*/1, /*Type*/GILLT_v8s1,
          GIM_CheckRegBankForClass, /*MI*/1, /*Op*/1, /*RC*/X86::VK8RegClassID,
          GIM_CheckIsSafeToFold, /*InsnID*/1,
          // (anyext:{ *:[i32] } (bitconvert:{ *:[i8] } VK8:{ *:[v8i1] }:$src))  =>  (COPY_TO_REGCLASS:{ *:[i32] } VK8:{ *:[v8i1] }:$src, GR32:{ *:[i32] })
          GIR_BuildMI, /*InsnID*/0, /*Opcode*/TargetOpcode::COPY,
          GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
          GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/1, /*OpIdx*/1, // src
          GIR_EraseFromParent, /*InsnID*/0,
          GIR_ConstrainOperandRC, /*InsnID*/0, /*Op*/0, /*RC GR32*/27,
          // GIR_Coverage, 13702,
          GIR_Done,
        // Label 639: @19797
        GIM_Try, /*On fail goto*//*Label 640*/ 19810, // Rule ID 15947 //
          GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::GR8RegClassID,
          // (anyext:{ *:[i32] } GR8:{ *:[i8] }:$src)  =>  (MOVZX32rr8:{ *:[i32] } GR8:{ *:[i8] }:$src)
          GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::MOVZX32rr8,
          GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
          // GIR_Coverage, 15947,
          GIR_Done,
        // Label 640: @19810
        GIM_Reject,
      // Label 638: @19811
      GIM_Reject,
      // Label 636: @19812
      GIM_Try, /*On fail goto*//*Label 641*/ 19853, // Rule ID 13700 //
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::GR32RegClassID,
        GIM_RecordInsn, /*DefineMI*/1, /*MI*/0, /*OpIdx*/1, // MIs[1]
        GIM_CheckOpcode, /*MI*/1, TargetOpcode::G_BITCAST,
        GIM_CheckType, /*MI*/1, /*Op*/1, /*Type*/GILLT_v16s1,
        GIM_CheckRegBankForClass, /*MI*/1, /*Op*/1, /*RC*/X86::VK16RegClassID,
        GIM_CheckIsSafeToFold, /*InsnID*/1,
        // (anyext:{ *:[i32] } (bitconvert:{ *:[i16] } VK16:{ *:[v16i1] }:$src))  =>  (COPY_TO_REGCLASS:{ *:[i32] } VK16:{ *:[v16i1] }:$src, GR32:{ *:[i32] })
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/TargetOpcode::COPY,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/1, /*OpIdx*/1, // src
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainOperandRC, /*InsnID*/0, /*Op*/0, /*RC GR32*/27,
        // GIR_Coverage, 13700,
        GIR_Done,
      // Label 641: @19853
      GIM_Reject,
      // Label 637: @19854
      GIM_Reject,
    // Label 634: @19855
    GIM_Reject,
    // Label 10: @19856
    GIM_SwitchType, /*MI*/0, /*Op*/0, /*[*/1, 22, /*)*//*default:*//*Label 650*/ 20190,
    /*GILLT_s8*//*Label 642*/ 19883,
    /*GILLT_s16*//*Label 643*/ 19962, 0, 0, 0, 0, 0, 0, 0,
    /*GILLT_v4s32*//*Label 644*/ 20000, 0, 0,
    /*GILLT_v8s16*//*Label 645*/ 20024,
    /*GILLT_v8s32*//*Label 646*/ 20071, 0, 0,
    /*GILLT_v16s8*//*Label 647*/ 20095,
    /*GILLT_v16s16*//*Label 648*/ 20142, 0, 0,
    /*GILLT_v32s8*//*Label 649*/ 20166,
    // Label 642: @19883
    GIM_Try, /*On fail goto*//*Label 651*/ 19922, // Rule ID 15995 //
      GIM_CheckFeatures, GIFBS_In64BitMode,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_s32,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::GR32RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::GR32RegClassID,
      // (trunc:{ *:[i8] } GR32:{ *:[i32] }:$src)  =>  (EXTRACT_SUBREG:{ *:[i8] } GR32:{ *:[i32] }:$src, sub_8bit:{ *:[i32] })
      GIR_BuildMI, /*InsnID*/0, /*Opcode*/TargetOpcode::COPY,
      GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
      GIR_CopySubReg, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, /*SubRegIdx*/1, // src
      GIR_EraseFromParent, /*InsnID*/0,
      GIR_ConstrainOperandRC, /*InsnID*/0, /*Op*/0, /*RC GR8*/0,
      GIR_ConstrainOperandRC, /*InsnID*/0, /*Op*/1, /*RC GR32*/27,
      // GIR_Coverage, 15995,
      GIR_Done,
    // Label 651: @19922
    GIM_Try, /*On fail goto*//*Label 652*/ 19961, // Rule ID 15996 //
      GIM_CheckFeatures, GIFBS_In64BitMode,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_s16,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::GR16RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::GR16RegClassID,
      // (trunc:{ *:[i8] } GR16:{ *:[i16] }:$src)  =>  (EXTRACT_SUBREG:{ *:[i8] } GR16:{ *:[i16] }:$src, sub_8bit:{ *:[i32] })
      GIR_BuildMI, /*InsnID*/0, /*Opcode*/TargetOpcode::COPY,
      GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
      GIR_CopySubReg, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, /*SubRegIdx*/1, // src
      GIR_EraseFromParent, /*InsnID*/0,
      GIR_ConstrainOperandRC, /*InsnID*/0, /*Op*/0, /*RC GR8*/0,
      GIR_ConstrainOperandRC, /*InsnID*/0, /*Op*/1, /*RC GR16*/6,
      // GIR_Coverage, 15996,
      GIR_Done,
    // Label 652: @19961
    GIM_Reject,
    // Label 643: @19962
    GIM_Try, /*On fail goto*//*Label 653*/ 19999, // Rule ID 15989 //
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_s32,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::GR32RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::GR32RegClassID,
      // (trunc:{ *:[i16] } GR32:{ *:[i32] }:$src)  =>  (EXTRACT_SUBREG:{ *:[i16] } GR32:{ *:[i32] }:$src, sub_16bit:{ *:[i32] })
      GIR_BuildMI, /*InsnID*/0, /*Opcode*/TargetOpcode::COPY,
      GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
      GIR_CopySubReg, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, /*SubRegIdx*/4, // src
      GIR_EraseFromParent, /*InsnID*/0,
      GIR_ConstrainOperandRC, /*InsnID*/0, /*Op*/0, /*RC GR16*/6,
      GIR_ConstrainOperandRC, /*InsnID*/0, /*Op*/1, /*RC GR32*/27,
      // GIR_Coverage, 15989,
      GIR_Done,
    // Label 653: @19999
    GIM_Reject,
    // Label 644: @20000
    GIM_Try, /*On fail goto*//*Label 654*/ 20023, // Rule ID 9391 //
      GIM_CheckFeatures, GIFBS_HasAVX512_HasVLX,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v4s64,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128XRegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR256XRegClassID,
      // (trunc:{ *:[v4i32] } VR256X:{ *:[v4i64] }:$src1)  =>  (VPMOVQDZ256rr:{ *:[v4i32] } VR256X:{ *:[v4i64] }:$src1)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VPMOVQDZ256rr,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 9391,
      GIR_Done,
    // Label 654: @20023
    GIM_Reject,
    // Label 645: @20024
    GIM_Try, /*On fail goto*//*Label 655*/ 20047, // Rule ID 9367 //
      GIM_CheckFeatures, GIFBS_HasAVX512,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v8s64,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128XRegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR512RegClassID,
      // (trunc:{ *:[v8i16] } VR512:{ *:[v8i64] }:$src1)  =>  (VPMOVQWZrr:{ *:[v8i16] } VR512:{ *:[v8i64] }:$src1)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VPMOVQWZrr,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 9367,
      GIR_Done,
    // Label 655: @20047
    GIM_Try, /*On fail goto*//*Label 656*/ 20070, // Rule ID 9445 //
      GIM_CheckFeatures, GIFBS_HasAVX512_HasVLX,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v8s32,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128XRegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR256XRegClassID,
      // (trunc:{ *:[v8i16] } VR256X:{ *:[v8i32] }:$src1)  =>  (VPMOVDWZ256rr:{ *:[v8i16] } VR256X:{ *:[v8i32] }:$src1)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VPMOVDWZ256rr,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 9445,
      GIR_Done,
    // Label 656: @20070
    GIM_Reject,
    // Label 646: @20071
    GIM_Try, /*On fail goto*//*Label 657*/ 20094, // Rule ID 9394 //
      GIM_CheckFeatures, GIFBS_HasAVX512,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v8s64,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR256XRegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR512RegClassID,
      // (trunc:{ *:[v8i32] } VR512:{ *:[v8i64] }:$src1)  =>  (VPMOVQDZrr:{ *:[v8i32] } VR512:{ *:[v8i64] }:$src1)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VPMOVQDZrr,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 9394,
      GIR_Done,
    // Label 657: @20094
    GIM_Reject,
    // Label 647: @20095
    GIM_Try, /*On fail goto*//*Label 658*/ 20118, // Rule ID 9421 //
      GIM_CheckFeatures, GIFBS_HasAVX512,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v16s32,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128XRegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR512RegClassID,
      // (trunc:{ *:[v16i8] } VR512:{ *:[v16i32] }:$src1)  =>  (VPMOVDBZrr:{ *:[v16i8] } VR512:{ *:[v16i32] }:$src1)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VPMOVDBZrr,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 9421,
      GIR_Done,
    // Label 658: @20118
    GIM_Try, /*On fail goto*//*Label 659*/ 20141, // Rule ID 9472 //
      GIM_CheckFeatures, GIFBS_HasBWI_HasVLX,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v16s16,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128XRegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR256XRegClassID,
      // (trunc:{ *:[v16i8] } VR256X:{ *:[v16i16] }:$src1)  =>  (VPMOVWBZ256rr:{ *:[v16i8] } VR256X:{ *:[v16i16] }:$src1)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VPMOVWBZ256rr,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 9472,
      GIR_Done,
    // Label 659: @20141
    GIM_Reject,
    // Label 648: @20142
    GIM_Try, /*On fail goto*//*Label 660*/ 20165, // Rule ID 9448 //
      GIM_CheckFeatures, GIFBS_HasAVX512,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v16s32,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR256XRegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR512RegClassID,
      // (trunc:{ *:[v16i16] } VR512:{ *:[v16i32] }:$src1)  =>  (VPMOVDWZrr:{ *:[v16i16] } VR512:{ *:[v16i32] }:$src1)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VPMOVDWZrr,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 9448,
      GIR_Done,
    // Label 660: @20165
    GIM_Reject,
    // Label 649: @20166
    GIM_Try, /*On fail goto*//*Label 661*/ 20189, // Rule ID 9475 //
      GIM_CheckFeatures, GIFBS_HasBWI,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v32s16,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR256XRegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR512RegClassID,
      // (trunc:{ *:[v32i8] } VR512:{ *:[v32i16] }:$src1)  =>  (VPMOVWBZrr:{ *:[v32i8] } VR512:{ *:[v32i16] }:$src1)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VPMOVWBZrr,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 9475,
      GIR_Done,
    // Label 661: @20189
    GIM_Reject,
    // Label 650: @20190
    GIM_Reject,
    // Label 11: @20191
    GIM_SwitchType, /*MI*/0, /*Op*/0, /*[*/1, 5, /*)*//*default:*//*Label 666*/ 20393,
    /*GILLT_s8*//*Label 662*/ 20201,
    /*GILLT_s16*//*Label 663*/ 20223,
    /*GILLT_s32*//*Label 664*/ 20245,
    /*GILLT_s64*//*Label 665*/ 20342,
    // Label 662: @20201
    GIM_Try, /*On fail goto*//*Label 667*/ 20222, // Rule ID 19 //
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::GR8RegClassID,
      // MIs[0] Operand 1
      // No operand predicates
      // (imm:{ *:[i8] }):$src  =>  (MOV8ri:{ *:[i8] } (imm:{ *:[i8] }):$src)
      GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::MOV8ri,
      GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
      GIR_CopyConstantAsSImm, /*NewInsnID*/0, /*OldInsnID*/0, // src
      GIR_EraseFromParent, /*InsnID*/0,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 19,
      GIR_Done,
    // Label 667: @20222
    GIM_Reject,
    // Label 663: @20223
    GIM_Try, /*On fail goto*//*Label 668*/ 20244, // Rule ID 20 //
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::GR16RegClassID,
      // MIs[0] Operand 1
      // No operand predicates
      // (imm:{ *:[i16] }):$src  =>  (MOV16ri:{ *:[i16] } (imm:{ *:[i16] }):$src)
      GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::MOV16ri,
      GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
      GIR_CopyConstantAsSImm, /*NewInsnID*/0, /*OldInsnID*/0, // src
      GIR_EraseFromParent, /*InsnID*/0,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 20,
      GIR_Done,
    // Label 668: @20244
    GIM_Reject,
    // Label 664: @20245
    GIM_Try, /*On fail goto*//*Label 669*/ 20267, // Rule ID 12065 //
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::GR32RegClassID,
      // MIs[0] Operand 1
      GIM_CheckLiteralInt, /*MI*/0, /*Op*/1, 0,
      // 0:{ *:[i32] }  =>  (MOV32r0:{ *:[i32] }:{ *:[i32] })
      GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::MOV32r0,
      GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
      GIR_EraseFromParent, /*InsnID*/0,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 12065,
      GIR_Done,
    // Label 669: @20267
    GIM_Try, /*On fail goto*//*Label 670*/ 20291, // Rule ID 12066 //
      GIM_CheckFeatures, GIFBS_Not64BitMode_OptForSize,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::GR32RegClassID,
      // MIs[0] Operand 1
      GIM_CheckLiteralInt, /*MI*/0, /*Op*/1, 1,
      // 1:{ *:[i32] }  =>  (MOV32r1:{ *:[i32] }:{ *:[i32] })
      GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::MOV32r1,
      GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
      GIR_EraseFromParent, /*InsnID*/0,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 12066,
      GIR_Done,
    // Label 670: @20291
    GIM_Try, /*On fail goto*//*Label 671*/ 20315, // Rule ID 12067 //
      GIM_CheckFeatures, GIFBS_Not64BitMode_OptForSize,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::GR32RegClassID,
      // MIs[0] Operand 1
      GIM_CheckLiteralInt, /*MI*/0, /*Op*/1, -1,
      // -1:{ *:[i32] }  =>  (MOV32r_1:{ *:[i32] }:{ *:[i32] })
      GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::MOV32r_1,
      GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
      GIR_EraseFromParent, /*InsnID*/0,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 12067,
      GIR_Done,
    // Label 671: @20315
    GIM_Try, /*On fail goto*//*Label 672*/ 20341, // Rule ID 12068 //
      GIM_CheckFeatures, GIFBS_NotWin64WithoutFP_OptForMinSize,
      GIM_CheckI64ImmPredicate, /*MI*/0, /*Predicate*/GIPFP_I64_Predicate_i32immSExt8,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::GR32RegClassID,
      // MIs[0] Operand 1
      // No operand predicates
      // (imm:{ *:[i32] })<<P:Predicate_i32immSExt8>>:$src  =>  (MOV32ImmSExti8:{ *:[i32] } (imm:{ *:[i32] }):$src)
      GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::MOV32ImmSExti8,
      GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
      GIR_CopyConstantAsSImm, /*NewInsnID*/0, /*OldInsnID*/0, // src
      GIR_EraseFromParent, /*InsnID*/0,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 12068,
      GIR_Done,
    // Label 672: @20341
    GIM_Reject,
    // Label 665: @20342
    GIM_Try, /*On fail goto*//*Label 673*/ 20368, // Rule ID 12069 //
      GIM_CheckFeatures, GIFBS_NotWin64WithoutFP_OptForMinSize,
      GIM_CheckI64ImmPredicate, /*MI*/0, /*Predicate*/GIPFP_I64_Predicate_i64immSExt8,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::GR64RegClassID,
      // MIs[0] Operand 1
      // No operand predicates
      // (imm:{ *:[i64] })<<P:Predicate_i64immSExt8>>:$src  =>  (MOV64ImmSExti8:{ *:[i64] } (imm:{ *:[i64] }):$src)
      GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::MOV64ImmSExti8,
      GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
      GIR_CopyConstantAsSImm, /*NewInsnID*/0, /*OldInsnID*/0, // src
      GIR_EraseFromParent, /*InsnID*/0,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 12069,
      GIR_Done,
    // Label 673: @20368
    GIM_Try, /*On fail goto*//*Label 674*/ 20392, // Rule ID 22 //
      GIM_CheckI64ImmPredicate, /*MI*/0, /*Predicate*/GIPFP_I64_Predicate_i64immSExt32,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::GR64RegClassID,
      // MIs[0] Operand 1
      // No operand predicates
      // (imm:{ *:[i64] })<<P:Predicate_i64immSExt32>>:$src  =>  (MOV64ri32:{ *:[i64] } (imm:{ *:[i64] }):$src)
      GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::MOV64ri32,
      GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
      GIR_CopyConstantAsSImm, /*NewInsnID*/0, /*OldInsnID*/0, // src
      GIR_EraseFromParent, /*InsnID*/0,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 22,
      GIR_Done,
    // Label 674: @20392
    GIM_Reject,
    // Label 666: @20393
    GIM_Reject,
    // Label 12: @20394
    GIM_SwitchType, /*MI*/0, /*Op*/0, /*[*/3, 6, /*)*//*default:*//*Label 678*/ 20586,
    /*GILLT_s32*//*Label 675*/ 20403,
    /*GILLT_s64*//*Label 676*/ 20450,
    /*GILLT_s80*//*Label 677*/ 20543,
    // Label 675: @20403
    GIM_Try, /*On fail goto*//*Label 679*/ 20426, // Rule ID 887 //
      GIM_CheckFeatures, GIFBS_FPStackf32,
      GIM_CheckAPFloatImmPredicate, /*MI*/0, /*Predicate*/GIPFP_APFloat_Predicate_fpimm0,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::RFP32RegClassID,
      // MIs[0] Operand 1
      // No operand predicates
      // (fpimm:{ *:[f32] })<<P:Predicate_fpimm0>>  =>  (LD_Fp032:{ *:[f32] }:{ *:[i16] })
      GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::LD_Fp032,
      GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
      GIR_EraseFromParent, /*InsnID*/0,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 887,
      GIR_Done,
    // Label 679: @20426
    GIM_Try, /*On fail goto*//*Label 680*/ 20449, // Rule ID 888 //
      GIM_CheckFeatures, GIFBS_FPStackf32,
      GIM_CheckAPFloatImmPredicate, /*MI*/0, /*Predicate*/GIPFP_APFloat_Predicate_fpimm1,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::RFP32RegClassID,
      // MIs[0] Operand 1
      // No operand predicates
      // (fpimm:{ *:[f32] })<<P:Predicate_fpimm1>>  =>  (LD_Fp132:{ *:[f32] }:{ *:[i16] })
      GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::LD_Fp132,
      GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
      GIR_EraseFromParent, /*InsnID*/0,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 888,
      GIR_Done,
    // Label 680: @20449
    GIM_Reject,
    // Label 676: @20450
    GIM_Try, /*On fail goto*//*Label 681*/ 20473, // Rule ID 889 //
      GIM_CheckFeatures, GIFBS_FPStackf64,
      GIM_CheckAPFloatImmPredicate, /*MI*/0, /*Predicate*/GIPFP_APFloat_Predicate_fpimm0,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::RFP64RegClassID,
      // MIs[0] Operand 1
      // No operand predicates
      // (fpimm:{ *:[f64] })<<P:Predicate_fpimm0>>  =>  (LD_Fp064:{ *:[f64] }:{ *:[i16] })
      GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::LD_Fp064,
      GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
      GIR_EraseFromParent, /*InsnID*/0,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 889,
      GIR_Done,
    // Label 681: @20473
    GIM_Try, /*On fail goto*//*Label 682*/ 20496, // Rule ID 890 //
      GIM_CheckFeatures, GIFBS_FPStackf64,
      GIM_CheckAPFloatImmPredicate, /*MI*/0, /*Predicate*/GIPFP_APFloat_Predicate_fpimm1,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::RFP64RegClassID,
      // MIs[0] Operand 1
      // No operand predicates
      // (fpimm:{ *:[f64] })<<P:Predicate_fpimm1>>  =>  (LD_Fp164:{ *:[f64] }:{ *:[i16] })
      GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::LD_Fp164,
      GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
      GIR_EraseFromParent, /*InsnID*/0,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 890,
      GIR_Done,
    // Label 682: @20496
    GIM_Try, /*On fail goto*//*Label 683*/ 20519, // Rule ID 1277 //
      GIM_CheckFeatures, GIFBS_HasSSE2_NoAVX512,
      GIM_CheckAPFloatImmPredicate, /*MI*/0, /*Predicate*/GIPFP_APFloat_Predicate_fpimm0,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::FR64RegClassID,
      // MIs[0] Operand 1
      // No operand predicates
      // (fpimm:{ *:[f64] })<<P:Predicate_fpimm0>>  =>  (FsFLD0SD:{ *:[f64] })
      GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::FsFLD0SD,
      GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
      GIR_EraseFromParent, /*InsnID*/0,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 1277,
      GIR_Done,
    // Label 683: @20519
    GIM_Try, /*On fail goto*//*Label 684*/ 20542, // Rule ID 2780 //
      GIM_CheckFeatures, GIFBS_HasAVX512,
      GIM_CheckAPFloatImmPredicate, /*MI*/0, /*Predicate*/GIPFP_APFloat_Predicate_fpimm0,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::FR64XRegClassID,
      // MIs[0] Operand 1
      // No operand predicates
      // (fpimm:{ *:[f64] })<<P:Predicate_fpimm0>>  =>  (AVX512_FsFLD0SD:{ *:[f64] })
      GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::AVX512_FsFLD0SD,
      GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
      GIR_EraseFromParent, /*InsnID*/0,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 2780,
      GIR_Done,
    // Label 684: @20542
    GIM_Reject,
    // Label 677: @20543
    GIM_Try, /*On fail goto*//*Label 685*/ 20564, // Rule ID 891 //
      GIM_CheckAPFloatImmPredicate, /*MI*/0, /*Predicate*/GIPFP_APFloat_Predicate_fpimm0,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::RFP80RegClassID,
      // MIs[0] Operand 1
      // No operand predicates
      // (fpimm:{ *:[f80] })<<P:Predicate_fpimm0>>  =>  (LD_Fp080:{ *:[f80] }:{ *:[i16] })
      GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::LD_Fp080,
      GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
      GIR_EraseFromParent, /*InsnID*/0,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 891,
      GIR_Done,
    // Label 685: @20564
    GIM_Try, /*On fail goto*//*Label 686*/ 20585, // Rule ID 892 //
      GIM_CheckAPFloatImmPredicate, /*MI*/0, /*Predicate*/GIPFP_APFloat_Predicate_fpimm1,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::RFP80RegClassID,
      // MIs[0] Operand 1
      // No operand predicates
      // (fpimm:{ *:[f80] })<<P:Predicate_fpimm1>>  =>  (LD_Fp180:{ *:[f80] }:{ *:[i16] })
      GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::LD_Fp180,
      GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
      GIR_EraseFromParent, /*InsnID*/0,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 892,
      GIR_Done,
    // Label 686: @20585
    GIM_Reject,
    // Label 678: @20586
    GIM_Reject,
    // Label 13: @20587
    GIM_SwitchType, /*MI*/0, /*Op*/0, /*[*/3, 25, /*)*//*default:*//*Label 701*/ 21129,
    /*GILLT_s32*//*Label 687*/ 20615,
    /*GILLT_s64*//*Label 688*/ 20658, 0, 0, 0,
    /*GILLT_v2s64*//*Label 689*/ 20724, 0,
    /*GILLT_v4s32*//*Label 690*/ 20748,
    /*GILLT_v4s64*//*Label 691*/ 20772, 0,
    /*GILLT_v8s16*//*Label 692*/ 20796,
    /*GILLT_v8s32*//*Label 693*/ 20859,
    /*GILLT_v8s64*//*Label 694*/ 20883, 0,
    /*GILLT_v16s8*//*Label 695*/ 20907,
    /*GILLT_v16s16*//*Label 696*/ 20970,
    /*GILLT_v16s32*//*Label 697*/ 21033, 0,
    /*GILLT_v32s8*//*Label 698*/ 21057,
    /*GILLT_v32s16*//*Label 699*/ 21081, 0,
    /*GILLT_v64s8*//*Label 700*/ 21105,
    // Label 687: @20615
    GIM_Try, /*On fail goto*//*Label 702*/ 20636, // Rule ID 532 //
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_s8,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::GR32RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::GR8RegClassID,
      // (sext:{ *:[i32] } GR8:{ *:[i8] }:$src)  =>  (MOVSX32rr8:{ *:[i32] } GR8:{ *:[i8] }:$src)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::MOVSX32rr8,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 532,
      GIR_Done,
    // Label 702: @20636
    GIM_Try, /*On fail goto*//*Label 703*/ 20657, // Rule ID 534 //
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_s16,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::GR32RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::GR16RegClassID,
      // (sext:{ *:[i32] } GR16:{ *:[i16] }:$src)  =>  (MOVSX32rr16:{ *:[i32] } GR16:{ *:[i16] }:$src)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::MOVSX32rr16,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 534,
      GIR_Done,
    // Label 703: @20657
    GIM_Reject,
    // Label 688: @20658
    GIM_Try, /*On fail goto*//*Label 704*/ 20679, // Rule ID 540 //
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_s8,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::GR64RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::GR8RegClassID,
      // (sext:{ *:[i64] } GR8:{ *:[i8] }:$src)  =>  (MOVSX64rr8:{ *:[i64] } GR8:{ *:[i8] }:$src)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::MOVSX64rr8,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 540,
      GIR_Done,
    // Label 704: @20679
    GIM_Try, /*On fail goto*//*Label 705*/ 20700, // Rule ID 542 //
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_s16,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::GR64RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::GR16RegClassID,
      // (sext:{ *:[i64] } GR16:{ *:[i16] }:$src)  =>  (MOVSX64rr16:{ *:[i64] } GR16:{ *:[i16] }:$src)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::MOVSX64rr16,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 542,
      GIR_Done,
    // Label 705: @20700
    GIM_Try, /*On fail goto*//*Label 706*/ 20723, // Rule ID 544 //
      GIM_CheckFeatures, GIFBS_In64BitMode,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_s32,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::GR64RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::GR32RegClassID,
      // (sext:{ *:[i64] } GR32:{ *:[i32] }:$src)  =>  (MOVSX64rr32:{ *:[i64] } GR32:{ *:[i32] }:$src)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::MOVSX64rr32,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 544,
      GIR_Done,
    // Label 706: @20723
    GIM_Reject,
    // Label 689: @20724
    GIM_Try, /*On fail goto*//*Label 707*/ 20747, // Rule ID 9771 //
      GIM_CheckFeatures, GIFBS_HasDQI_HasVLX,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v2s1,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128XRegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VK2RegClassID,
      // (sext:{ *:[v2i64] } VK2:{ *:[v2i1] }:$src)  =>  (VPMOVM2QZ128rr:{ *:[v2i64] } VK2:{ *:[v2i1] }:$src)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VPMOVM2QZ128rr,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 9771,
      GIR_Done,
    // Label 707: @20747
    GIM_Reject,
    // Label 690: @20748
    GIM_Try, /*On fail goto*//*Label 708*/ 20771, // Rule ID 9768 //
      GIM_CheckFeatures, GIFBS_HasDQI_HasVLX,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v4s1,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128XRegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VK4RegClassID,
      // (sext:{ *:[v4i32] } VK4:{ *:[v4i1] }:$src)  =>  (VPMOVM2DZ128rr:{ *:[v4i32] } VK4:{ *:[v4i1] }:$src)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VPMOVM2DZ128rr,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 9768,
      GIR_Done,
    // Label 708: @20771
    GIM_Reject,
    // Label 691: @20772
    GIM_Try, /*On fail goto*//*Label 709*/ 20795, // Rule ID 9770 //
      GIM_CheckFeatures, GIFBS_HasDQI_HasVLX,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v4s1,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR256XRegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VK4RegClassID,
      // (sext:{ *:[v4i64] } VK4:{ *:[v4i1] }:$src)  =>  (VPMOVM2QZ256rr:{ *:[v4i64] } VK4:{ *:[v4i1] }:$src)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VPMOVM2QZ256rr,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 9770,
      GIR_Done,
    // Label 709: @20795
    GIM_Reject,
    // Label 692: @20796
    GIM_Try, /*On fail goto*//*Label 710*/ 20858,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v8s1,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128XRegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VK8RegClassID,
      GIM_Try, /*On fail goto*//*Label 711*/ 20821, // Rule ID 9765 //
        GIM_CheckFeatures, GIFBS_HasBWI_HasVLX,
        // (sext:{ *:[v8i16] } VK8:{ *:[v8i1] }:$src)  =>  (VPMOVM2WZ128rr:{ *:[v8i16] } VK8:{ *:[v8i1] }:$src)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VPMOVM2WZ128rr,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 9765,
        GIR_Done,
      // Label 711: @20821
      GIM_Try, /*On fail goto*//*Label 712*/ 20857, // Rule ID 15174 //
        GIM_CheckFeatures, GIFBS_HasDQI_HasVLX_NoBWI,
        // (sext:{ *:[v8i16] } VK8:{ *:[v8i1] }:$src)  =>  (VPMOVDWZ256rr:{ *:[v8i16] } (VPMOVM2DZ256rr:{ *:[v8i32] } VK8:{ *:[v8i1] }:$src))
        GIR_MakeTempReg, /*TempRegID*/0, /*TypeID*/GILLT_v8s32,
        GIR_BuildMI, /*InsnID*/1, /*Opcode*/X86::VPMOVM2DZ256rr,
        GIR_AddTempRegister, /*InsnID*/1, /*TempRegID*/0, /*TempRegFlags*/RegState::Define,
        GIR_Copy, /*NewInsnID*/1, /*OldInsnID*/0, /*OpIdx*/1, // src
        GIR_ConstrainSelectedInstOperands, /*InsnID*/1,
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::VPMOVDWZ256rr,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_AddTempRegister, /*InsnID*/0, /*TempRegID*/0, /*TempRegFlags*/0,
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 15174,
        GIR_Done,
      // Label 712: @20857
      GIM_Reject,
    // Label 710: @20858
    GIM_Reject,
    // Label 693: @20859
    GIM_Try, /*On fail goto*//*Label 713*/ 20882, // Rule ID 9767 //
      GIM_CheckFeatures, GIFBS_HasDQI_HasVLX,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v8s1,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR256XRegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VK8RegClassID,
      // (sext:{ *:[v8i32] } VK8:{ *:[v8i1] }:$src)  =>  (VPMOVM2DZ256rr:{ *:[v8i32] } VK8:{ *:[v8i1] }:$src)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VPMOVM2DZ256rr,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 9767,
      GIR_Done,
    // Label 713: @20882
    GIM_Reject,
    // Label 694: @20883
    GIM_Try, /*On fail goto*//*Label 714*/ 20906, // Rule ID 9769 //
      GIM_CheckFeatures, GIFBS_HasDQI,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v8s1,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR512RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VK8RegClassID,
      // (sext:{ *:[v8i64] } VK8:{ *:[v8i1] }:$src)  =>  (VPMOVM2QZrr:{ *:[v8i64] } VK8:{ *:[v8i1] }:$src)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VPMOVM2QZrr,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 9769,
      GIR_Done,
    // Label 714: @20906
    GIM_Reject,
    // Label 695: @20907
    GIM_Try, /*On fail goto*//*Label 715*/ 20969,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v16s1,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128XRegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VK16RegClassID,
      GIM_Try, /*On fail goto*//*Label 716*/ 20932, // Rule ID 9762 //
        GIM_CheckFeatures, GIFBS_HasBWI_HasVLX,
        // (sext:{ *:[v16i8] } VK16:{ *:[v16i1] }:$src)  =>  (VPMOVM2BZ128rr:{ *:[v16i8] } VK16:{ *:[v16i1] }:$src)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VPMOVM2BZ128rr,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 9762,
        GIR_Done,
      // Label 716: @20932
      GIM_Try, /*On fail goto*//*Label 717*/ 20968, // Rule ID 15172 //
        GIM_CheckFeatures, GIFBS_HasDQI_NoBWI,
        // (sext:{ *:[v16i8] } VK16:{ *:[v16i1] }:$src)  =>  (VPMOVDBZrr:{ *:[v16i8] } (VPMOVM2DZrr:{ *:[v16i32] } VK16:{ *:[v16i1] }:$src))
        GIR_MakeTempReg, /*TempRegID*/0, /*TypeID*/GILLT_v16s32,
        GIR_BuildMI, /*InsnID*/1, /*Opcode*/X86::VPMOVM2DZrr,
        GIR_AddTempRegister, /*InsnID*/1, /*TempRegID*/0, /*TempRegFlags*/RegState::Define,
        GIR_Copy, /*NewInsnID*/1, /*OldInsnID*/0, /*OpIdx*/1, // src
        GIR_ConstrainSelectedInstOperands, /*InsnID*/1,
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::VPMOVDBZrr,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_AddTempRegister, /*InsnID*/0, /*TempRegID*/0, /*TempRegFlags*/0,
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 15172,
        GIR_Done,
      // Label 717: @20968
      GIM_Reject,
    // Label 715: @20969
    GIM_Reject,
    // Label 696: @20970
    GIM_Try, /*On fail goto*//*Label 718*/ 21032,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v16s1,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR256XRegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VK16RegClassID,
      GIM_Try, /*On fail goto*//*Label 719*/ 20995, // Rule ID 9764 //
        GIM_CheckFeatures, GIFBS_HasBWI_HasVLX,
        // (sext:{ *:[v16i16] } VK16:{ *:[v16i1] }:$src)  =>  (VPMOVM2WZ256rr:{ *:[v16i16] } VK16:{ *:[v16i1] }:$src)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VPMOVM2WZ256rr,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 9764,
        GIR_Done,
      // Label 719: @20995
      GIM_Try, /*On fail goto*//*Label 720*/ 21031, // Rule ID 15173 //
        GIM_CheckFeatures, GIFBS_HasDQI_NoBWI,
        // (sext:{ *:[v16i16] } VK16:{ *:[v16i1] }:$src)  =>  (VPMOVDWZrr:{ *:[v16i16] } (VPMOVM2DZrr:{ *:[v16i32] } VK16:{ *:[v16i1] }:$src))
        GIR_MakeTempReg, /*TempRegID*/0, /*TypeID*/GILLT_v16s32,
        GIR_BuildMI, /*InsnID*/1, /*Opcode*/X86::VPMOVM2DZrr,
        GIR_AddTempRegister, /*InsnID*/1, /*TempRegID*/0, /*TempRegFlags*/RegState::Define,
        GIR_Copy, /*NewInsnID*/1, /*OldInsnID*/0, /*OpIdx*/1, // src
        GIR_ConstrainSelectedInstOperands, /*InsnID*/1,
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::VPMOVDWZrr,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_AddTempRegister, /*InsnID*/0, /*TempRegID*/0, /*TempRegFlags*/0,
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 15173,
        GIR_Done,
      // Label 720: @21031
      GIM_Reject,
    // Label 718: @21032
    GIM_Reject,
    // Label 697: @21033
    GIM_Try, /*On fail goto*//*Label 721*/ 21056, // Rule ID 9766 //
      GIM_CheckFeatures, GIFBS_HasDQI,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v16s1,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR512RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VK16RegClassID,
      // (sext:{ *:[v16i32] } VK16:{ *:[v16i1] }:$src)  =>  (VPMOVM2DZrr:{ *:[v16i32] } VK16:{ *:[v16i1] }:$src)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VPMOVM2DZrr,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 9766,
      GIR_Done,
    // Label 721: @21056
    GIM_Reject,
    // Label 698: @21057
    GIM_Try, /*On fail goto*//*Label 722*/ 21080, // Rule ID 9761 //
      GIM_CheckFeatures, GIFBS_HasBWI_HasVLX,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v32s1,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR256XRegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VK32RegClassID,
      // (sext:{ *:[v32i8] } VK32:{ *:[v32i1] }:$src)  =>  (VPMOVM2BZ256rr:{ *:[v32i8] } VK32:{ *:[v32i1] }:$src)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VPMOVM2BZ256rr,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 9761,
      GIR_Done,
    // Label 722: @21080
    GIM_Reject,
    // Label 699: @21081
    GIM_Try, /*On fail goto*//*Label 723*/ 21104, // Rule ID 9763 //
      GIM_CheckFeatures, GIFBS_HasBWI,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v32s1,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR512RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VK32RegClassID,
      // (sext:{ *:[v32i16] } VK32:{ *:[v32i1] }:$src)  =>  (VPMOVM2WZrr:{ *:[v32i16] } VK32:{ *:[v32i1] }:$src)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VPMOVM2WZrr,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 9763,
      GIR_Done,
    // Label 723: @21104
    GIM_Reject,
    // Label 700: @21105
    GIM_Try, /*On fail goto*//*Label 724*/ 21128, // Rule ID 9760 //
      GIM_CheckFeatures, GIFBS_HasBWI,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v64s1,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR512RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VK64RegClassID,
      // (sext:{ *:[v64i8] } VK64:{ *:[v64i1] }:$src)  =>  (VPMOVM2BZrr:{ *:[v64i8] } VK64:{ *:[v64i1] }:$src)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VPMOVM2BZrr,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 9760,
      GIR_Done,
    // Label 724: @21128
    GIM_Reject,
    // Label 701: @21129
    GIM_Reject,
    // Label 14: @21130
    GIM_Try, /*On fail goto*//*Label 725*/ 21264,
      GIM_CheckType, /*MI*/0, /*Op*/0, /*Type*/GILLT_s32,
      GIM_SwitchType, /*MI*/0, /*Op*/1, /*[*/1, 3, /*)*//*default:*//*Label 728*/ 21242,
      /*GILLT_s8*//*Label 726*/ 21144,
      /*GILLT_s16*//*Label 727*/ 21202,
      // Label 726: @21144
      GIM_Try, /*On fail goto*//*Label 729*/ 21201,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::GR32RegClassID,
        GIM_Try, /*On fail goto*//*Label 730*/ 21187, // Rule ID 13701 //
          GIM_CheckFeatures, GIFBS_HasDQI,
          GIM_RecordInsn, /*DefineMI*/1, /*MI*/0, /*OpIdx*/1, // MIs[1]
          GIM_CheckOpcode, /*MI*/1, TargetOpcode::G_BITCAST,
          GIM_CheckType, /*MI*/1, /*Op*/1, /*Type*/GILLT_v8s1,
          GIM_CheckRegBankForClass, /*MI*/1, /*Op*/1, /*RC*/X86::VK8RegClassID,
          GIM_CheckIsSafeToFold, /*InsnID*/1,
          // (zext:{ *:[i32] } (bitconvert:{ *:[i8] } VK8:{ *:[v8i1] }:$src))  =>  (KMOVBrk:{ *:[i32] } VK8:{ *:[v8i1] }:$src)
          GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::KMOVBrk,
          GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
          GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/1, /*OpIdx*/1, // src
          GIR_EraseFromParent, /*InsnID*/0,
          GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
          // GIR_Coverage, 13701,
          GIR_Done,
        // Label 730: @21187
        GIM_Try, /*On fail goto*//*Label 731*/ 21200, // Rule ID 536 //
          GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::GR8RegClassID,
          // (zext:{ *:[i32] } GR8:{ *:[i8] }:$src)  =>  (MOVZX32rr8:{ *:[i32] } GR8:{ *:[i8] }:$src)
          GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::MOVZX32rr8,
          GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
          // GIR_Coverage, 536,
          GIR_Done,
        // Label 731: @21200
        GIM_Reject,
      // Label 729: @21201
      GIM_Reject,
      // Label 727: @21202
      GIM_Try, /*On fail goto*//*Label 732*/ 21241, // Rule ID 13699 //
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::GR32RegClassID,
        GIM_RecordInsn, /*DefineMI*/1, /*MI*/0, /*OpIdx*/1, // MIs[1]
        GIM_CheckOpcode, /*MI*/1, TargetOpcode::G_BITCAST,
        GIM_CheckType, /*MI*/1, /*Op*/1, /*Type*/GILLT_v16s1,
        GIM_CheckRegBankForClass, /*MI*/1, /*Op*/1, /*RC*/X86::VK16RegClassID,
        GIM_CheckIsSafeToFold, /*InsnID*/1,
        // (zext:{ *:[i32] } (bitconvert:{ *:[i16] } VK16:{ *:[v16i1] }:$src))  =>  (KMOVWrk:{ *:[i32] } VK16:{ *:[v16i1] }:$src)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::KMOVWrk,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/1, /*OpIdx*/1, // src
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 13699,
        GIR_Done,
      // Label 732: @21241
      GIM_Reject,
      // Label 728: @21242
      GIM_Try, /*On fail goto*//*Label 733*/ 21263, // Rule ID 538 //
        GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_s16,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::GR32RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::GR16RegClassID,
        // (zext:{ *:[i32] } GR16:{ *:[i16] }:$src)  =>  (MOVZX32rr16:{ *:[i32] } GR16:{ *:[i16] }:$src)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::MOVZX32rr16,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 538,
        GIR_Done,
      // Label 733: @21263
      GIM_Reject,
    // Label 725: @21264
    GIM_Reject,
    // Label 15: @21265
    GIM_SwitchType, /*MI*/0, /*Op*/0, /*[*/1, 23, /*)*//*default:*//*Label 747*/ 21989,
    /*GILLT_s8*//*Label 734*/ 21293,
    /*GILLT_s16*//*Label 735*/ 21369,
    /*GILLT_s32*//*Label 736*/ 21445,
    /*GILLT_s64*//*Label 737*/ 21521, 0, 0, 0,
    /*GILLT_v2s64*//*Label 738*/ 21597, 0,
    /*GILLT_v4s32*//*Label 739*/ 21655,
    /*GILLT_v4s64*//*Label 740*/ 21713, 0,
    /*GILLT_v8s16*//*Label 741*/ 21771,
    /*GILLT_v8s32*//*Label 742*/ 21803,
    /*GILLT_v8s64*//*Label 743*/ 21861, 0, 0,
    /*GILLT_v16s16*//*Label 744*/ 21893,
    /*GILLT_v16s32*//*Label 745*/ 21925, 0, 0,
    /*GILLT_v32s16*//*Label 746*/ 21957,
    // Label 734: @21293
    GIM_Try, /*On fail goto*//*Label 748*/ 21368,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_s8,
      GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_s8,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::GR8RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::GR8RegClassID,
      GIM_Try, /*On fail goto*//*Label 749*/ 21337, // Rule ID 16011 //
        GIM_CheckConstantInt, /*MI*/0, /*Op*/2, 1,
        // (shl:{ *:[i8] } GR8:{ *:[i8] }:$src1, 1:{ *:[i8] })  =>  (ADD8rr:{ *:[i8] }:{ *:[i32] } GR8:{ *:[i8] }:$src1, GR8:{ *:[i8] }:$src1)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::ADD8rr,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // src1
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // src1
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 16011,
        GIR_Done,
      // Label 749: @21337
      GIM_Try, /*On fail goto*//*Label 750*/ 21367, // Rule ID 595 //
        GIM_RecordInsn, /*DefineMI*/1, /*MI*/0, /*OpIdx*/2, // MIs[1]
        GIM_CheckOpcode, /*MI*/1, TargetOpcode::G_CONSTANT,
        // MIs[1] Operand 1
        // No operand predicates
        GIM_CheckIsSafeToFold, /*InsnID*/1,
        // (shl:{ *:[i8] } GR8:{ *:[i8] }:$src1, (imm:{ *:[i8] }):$src2)  =>  (SHL8ri:{ *:[i8] }:{ *:[i32] } GR8:{ *:[i8] }:$src1, (imm:{ *:[i8] }):$src2)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::SHL8ri,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // src1
        GIR_CopyConstantAsSImm, /*NewInsnID*/0, /*OldInsnID*/1, // src2
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 595,
        GIR_Done,
      // Label 750: @21367
      GIM_Reject,
    // Label 748: @21368
    GIM_Reject,
    // Label 735: @21369
    GIM_Try, /*On fail goto*//*Label 751*/ 21444,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_s16,
      GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_s8,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::GR16RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::GR16RegClassID,
      GIM_Try, /*On fail goto*//*Label 752*/ 21413, // Rule ID 16012 //
        GIM_CheckConstantInt, /*MI*/0, /*Op*/2, 1,
        // (shl:{ *:[i16] } GR16:{ *:[i16] }:$src1, 1:{ *:[i8] })  =>  (ADD16rr:{ *:[i16] }:{ *:[i32] } GR16:{ *:[i16] }:$src1, GR16:{ *:[i16] }:$src1)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::ADD16rr,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // src1
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // src1
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 16012,
        GIR_Done,
      // Label 752: @21413
      GIM_Try, /*On fail goto*//*Label 753*/ 21443, // Rule ID 596 //
        GIM_RecordInsn, /*DefineMI*/1, /*MI*/0, /*OpIdx*/2, // MIs[1]
        GIM_CheckOpcode, /*MI*/1, TargetOpcode::G_CONSTANT,
        // MIs[1] Operand 1
        // No operand predicates
        GIM_CheckIsSafeToFold, /*InsnID*/1,
        // (shl:{ *:[i16] } GR16:{ *:[i16] }:$src1, (imm:{ *:[i8] }):$src2)  =>  (SHL16ri:{ *:[i16] }:{ *:[i32] } GR16:{ *:[i16] }:$src1, (imm:{ *:[i8] }):$src2)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::SHL16ri,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // src1
        GIR_CopyConstantAsSImm, /*NewInsnID*/0, /*OldInsnID*/1, // src2
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 596,
        GIR_Done,
      // Label 753: @21443
      GIM_Reject,
    // Label 751: @21444
    GIM_Reject,
    // Label 736: @21445
    GIM_Try, /*On fail goto*//*Label 754*/ 21520,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_s32,
      GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_s8,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::GR32RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::GR32RegClassID,
      GIM_Try, /*On fail goto*//*Label 755*/ 21489, // Rule ID 16013 //
        GIM_CheckConstantInt, /*MI*/0, /*Op*/2, 1,
        // (shl:{ *:[i32] } GR32:{ *:[i32] }:$src1, 1:{ *:[i8] })  =>  (ADD32rr:{ *:[i32] }:{ *:[i32] } GR32:{ *:[i32] }:$src1, GR32:{ *:[i32] }:$src1)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::ADD32rr,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // src1
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // src1
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 16013,
        GIR_Done,
      // Label 755: @21489
      GIM_Try, /*On fail goto*//*Label 756*/ 21519, // Rule ID 597 //
        GIM_RecordInsn, /*DefineMI*/1, /*MI*/0, /*OpIdx*/2, // MIs[1]
        GIM_CheckOpcode, /*MI*/1, TargetOpcode::G_CONSTANT,
        // MIs[1] Operand 1
        // No operand predicates
        GIM_CheckIsSafeToFold, /*InsnID*/1,
        // (shl:{ *:[i32] } GR32:{ *:[i32] }:$src1, (imm:{ *:[i8] }):$src2)  =>  (SHL32ri:{ *:[i32] }:{ *:[i32] } GR32:{ *:[i32] }:$src1, (imm:{ *:[i8] }):$src2)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::SHL32ri,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // src1
        GIR_CopyConstantAsSImm, /*NewInsnID*/0, /*OldInsnID*/1, // src2
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 597,
        GIR_Done,
      // Label 756: @21519
      GIM_Reject,
    // Label 754: @21520
    GIM_Reject,
    // Label 737: @21521
    GIM_Try, /*On fail goto*//*Label 757*/ 21596,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_s64,
      GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_s8,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::GR64RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::GR64RegClassID,
      GIM_Try, /*On fail goto*//*Label 758*/ 21565, // Rule ID 16014 //
        GIM_CheckConstantInt, /*MI*/0, /*Op*/2, 1,
        // (shl:{ *:[i64] } GR64:{ *:[i64] }:$src1, 1:{ *:[i8] })  =>  (ADD64rr:{ *:[i64] }:{ *:[i32] } GR64:{ *:[i64] }:$src1, GR64:{ *:[i64] }:$src1)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::ADD64rr,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // src1
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // src1
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 16014,
        GIR_Done,
      // Label 758: @21565
      GIM_Try, /*On fail goto*//*Label 759*/ 21595, // Rule ID 598 //
        GIM_RecordInsn, /*DefineMI*/1, /*MI*/0, /*OpIdx*/2, // MIs[1]
        GIM_CheckOpcode, /*MI*/1, TargetOpcode::G_CONSTANT,
        // MIs[1] Operand 1
        // No operand predicates
        GIM_CheckIsSafeToFold, /*InsnID*/1,
        // (shl:{ *:[i64] } GR64:{ *:[i64] }:$src1, (imm:{ *:[i8] }):$src2)  =>  (SHL64ri:{ *:[i64] }:{ *:[i32] } GR64:{ *:[i64] }:$src1, (imm:{ *:[i8] }):$src2)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::SHL64ri,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // src1
        GIR_CopyConstantAsSImm, /*NewInsnID*/0, /*OldInsnID*/1, // src2
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 598,
        GIR_Done,
      // Label 759: @21595
      GIM_Reject,
    // Label 757: @21596
    GIM_Reject,
    // Label 738: @21597
    GIM_Try, /*On fail goto*//*Label 760*/ 21654,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v2s64,
      GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v2s64,
      GIM_Try, /*On fail goto*//*Label 761*/ 21630, // Rule ID 2723 //
        GIM_CheckFeatures, GIFBS_HasAVX2_NoVLX,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR128RegClassID,
        // (shl:{ *:[v2i64] } VR128:{ *:[v2i64] }:$src1, VR128:{ *:[v2i64] }:$src2)  =>  (VPSLLVQrr:{ *:[v2i64] } VR128:{ *:[v2i64] }:$src1, VR128:{ *:[v2i64] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VPSLLVQrr,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 2723,
        GIR_Done,
      // Label 761: @21630
      GIM_Try, /*On fail goto*//*Label 762*/ 21653, // Rule ID 6397 //
        GIM_CheckFeatures, GIFBS_HasAVX512_HasVLX,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128XRegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR128XRegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR128XRegClassID,
        // (shl:{ *:[v2i64] } VR128X:{ *:[v2i64] }:$src1, VR128X:{ *:[v2i64] }:$src2)  =>  (VPSLLVQZ128rr:{ *:[v2i64] } VR128X:{ *:[v2i64] }:$src1, VR128X:{ *:[v2i64] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VPSLLVQZ128rr,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 6397,
        GIR_Done,
      // Label 762: @21653
      GIM_Reject,
    // Label 760: @21654
    GIM_Reject,
    // Label 739: @21655
    GIM_Try, /*On fail goto*//*Label 763*/ 21712,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v4s32,
      GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v4s32,
      GIM_Try, /*On fail goto*//*Label 764*/ 21688, // Rule ID 2719 //
        GIM_CheckFeatures, GIFBS_HasAVX2_NoVLX,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR128RegClassID,
        // (shl:{ *:[v4i32] } VR128:{ *:[v4i32] }:$src1, VR128:{ *:[v4i32] }:$src2)  =>  (VPSLLVDrr:{ *:[v4i32] } VR128:{ *:[v4i32] }:$src1, VR128:{ *:[v4i32] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VPSLLVDrr,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 2719,
        GIR_Done,
      // Label 764: @21688
      GIM_Try, /*On fail goto*//*Label 765*/ 21711, // Rule ID 6370 //
        GIM_CheckFeatures, GIFBS_HasAVX512_HasVLX,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128XRegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR128XRegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR128XRegClassID,
        // (shl:{ *:[v4i32] } VR128X:{ *:[v4i32] }:$src1, VR128X:{ *:[v4i32] }:$src2)  =>  (VPSLLVDZ128rr:{ *:[v4i32] } VR128X:{ *:[v4i32] }:$src1, VR128X:{ *:[v4i32] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VPSLLVDZ128rr,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 6370,
        GIR_Done,
      // Label 765: @21711
      GIM_Reject,
    // Label 763: @21712
    GIM_Reject,
    // Label 740: @21713
    GIM_Try, /*On fail goto*//*Label 766*/ 21770,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v4s64,
      GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v4s64,
      GIM_Try, /*On fail goto*//*Label 767*/ 21746, // Rule ID 2725 //
        GIM_CheckFeatures, GIFBS_HasAVX2_NoVLX,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR256RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR256RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR256RegClassID,
        // (shl:{ *:[v4i64] } VR256:{ *:[v4i64] }:$src1, VR256:{ *:[v4i64] }:$src2)  =>  (VPSLLVQYrr:{ *:[v4i64] } VR256:{ *:[v4i64] }:$src1, VR256:{ *:[v4i64] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VPSLLVQYrr,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 2725,
        GIR_Done,
      // Label 767: @21746
      GIM_Try, /*On fail goto*//*Label 768*/ 21769, // Rule ID 6388 //
        GIM_CheckFeatures, GIFBS_HasAVX512_HasVLX,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR256XRegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR256XRegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR256XRegClassID,
        // (shl:{ *:[v4i64] } VR256X:{ *:[v4i64] }:$src1, VR256X:{ *:[v4i64] }:$src2)  =>  (VPSLLVQZ256rr:{ *:[v4i64] } VR256X:{ *:[v4i64] }:$src1, VR256X:{ *:[v4i64] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VPSLLVQZ256rr,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 6388,
        GIR_Done,
      // Label 768: @21769
      GIM_Reject,
    // Label 766: @21770
    GIM_Reject,
    // Label 741: @21771
    GIM_Try, /*On fail goto*//*Label 769*/ 21802, // Rule ID 6418 //
      GIM_CheckFeatures, GIFBS_HasBWI_HasVLX,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v8s16,
      GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v8s16,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128XRegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR128XRegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR128XRegClassID,
      // (shl:{ *:[v8i16] } VR128X:{ *:[v8i16] }:$src1, VR128X:{ *:[v8i16] }:$src2)  =>  (VPSLLVWZ128rr:{ *:[v8i16] } VR128X:{ *:[v8i16] }:$src1, VR128X:{ *:[v8i16] }:$src2)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VPSLLVWZ128rr,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 6418,
      GIR_Done,
    // Label 769: @21802
    GIM_Reject,
    // Label 742: @21803
    GIM_Try, /*On fail goto*//*Label 770*/ 21860,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v8s32,
      GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v8s32,
      GIM_Try, /*On fail goto*//*Label 771*/ 21836, // Rule ID 2721 //
        GIM_CheckFeatures, GIFBS_HasAVX2_NoVLX,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR256RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR256RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR256RegClassID,
        // (shl:{ *:[v8i32] } VR256:{ *:[v8i32] }:$src1, VR256:{ *:[v8i32] }:$src2)  =>  (VPSLLVDYrr:{ *:[v8i32] } VR256:{ *:[v8i32] }:$src1, VR256:{ *:[v8i32] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VPSLLVDYrr,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 2721,
        GIR_Done,
      // Label 771: @21836
      GIM_Try, /*On fail goto*//*Label 772*/ 21859, // Rule ID 6361 //
        GIM_CheckFeatures, GIFBS_HasAVX512_HasVLX,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR256XRegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR256XRegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR256XRegClassID,
        // (shl:{ *:[v8i32] } VR256X:{ *:[v8i32] }:$src1, VR256X:{ *:[v8i32] }:$src2)  =>  (VPSLLVDZ256rr:{ *:[v8i32] } VR256X:{ *:[v8i32] }:$src1, VR256X:{ *:[v8i32] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VPSLLVDZ256rr,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 6361,
        GIR_Done,
      // Label 772: @21859
      GIM_Reject,
    // Label 770: @21860
    GIM_Reject,
    // Label 743: @21861
    GIM_Try, /*On fail goto*//*Label 773*/ 21892, // Rule ID 6379 //
      GIM_CheckFeatures, GIFBS_HasAVX512,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v8s64,
      GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v8s64,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR512RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR512RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR512RegClassID,
      // (shl:{ *:[v8i64] } VR512:{ *:[v8i64] }:$src1, VR512:{ *:[v8i64] }:$src2)  =>  (VPSLLVQZrr:{ *:[v8i64] } VR512:{ *:[v8i64] }:$src1, VR512:{ *:[v8i64] }:$src2)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VPSLLVQZrr,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 6379,
      GIR_Done,
    // Label 773: @21892
    GIM_Reject,
    // Label 744: @21893
    GIM_Try, /*On fail goto*//*Label 774*/ 21924, // Rule ID 6412 //
      GIM_CheckFeatures, GIFBS_HasBWI_HasVLX,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v16s16,
      GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v16s16,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR256XRegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR256XRegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR256XRegClassID,
      // (shl:{ *:[v16i16] } VR256X:{ *:[v16i16] }:$src1, VR256X:{ *:[v16i16] }:$src2)  =>  (VPSLLVWZ256rr:{ *:[v16i16] } VR256X:{ *:[v16i16] }:$src1, VR256X:{ *:[v16i16] }:$src2)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VPSLLVWZ256rr,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 6412,
      GIR_Done,
    // Label 774: @21924
    GIM_Reject,
    // Label 745: @21925
    GIM_Try, /*On fail goto*//*Label 775*/ 21956, // Rule ID 6352 //
      GIM_CheckFeatures, GIFBS_HasAVX512,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v16s32,
      GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v16s32,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR512RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR512RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR512RegClassID,
      // (shl:{ *:[v16i32] } VR512:{ *:[v16i32] }:$src1, VR512:{ *:[v16i32] }:$src2)  =>  (VPSLLVDZrr:{ *:[v16i32] } VR512:{ *:[v16i32] }:$src1, VR512:{ *:[v16i32] }:$src2)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VPSLLVDZrr,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 6352,
      GIR_Done,
    // Label 775: @21956
    GIM_Reject,
    // Label 746: @21957
    GIM_Try, /*On fail goto*//*Label 776*/ 21988, // Rule ID 6406 //
      GIM_CheckFeatures, GIFBS_HasBWI,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v32s16,
      GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v32s16,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR512RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR512RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR512RegClassID,
      // (shl:{ *:[v32i16] } VR512:{ *:[v32i16] }:$src1, VR512:{ *:[v32i16] }:$src2)  =>  (VPSLLVWZrr:{ *:[v32i16] } VR512:{ *:[v32i16] }:$src1, VR512:{ *:[v32i16] }:$src2)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VPSLLVWZrr,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 6406,
      GIR_Done,
    // Label 776: @21988
    GIM_Reject,
    // Label 747: @21989
    GIM_Reject,
    // Label 16: @21990
    GIM_SwitchType, /*MI*/0, /*Op*/0, /*[*/1, 23, /*)*//*default:*//*Label 790*/ 22820,
    /*GILLT_s8*//*Label 777*/ 22018,
    /*GILLT_s16*//*Label 778*/ 22090,
    /*GILLT_s32*//*Label 779*/ 22162,
    /*GILLT_s64*//*Label 780*/ 22356, 0, 0, 0,
    /*GILLT_v2s64*//*Label 781*/ 22428, 0,
    /*GILLT_v4s32*//*Label 782*/ 22486,
    /*GILLT_v4s64*//*Label 783*/ 22544, 0,
    /*GILLT_v8s16*//*Label 784*/ 22602,
    /*GILLT_v8s32*//*Label 785*/ 22634,
    /*GILLT_v8s64*//*Label 786*/ 22692, 0, 0,
    /*GILLT_v16s16*//*Label 787*/ 22724,
    /*GILLT_v16s32*//*Label 788*/ 22756, 0, 0,
    /*GILLT_v32s16*//*Label 789*/ 22788,
    // Label 777: @22018
    GIM_Try, /*On fail goto*//*Label 791*/ 22089,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_s8,
      GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_s8,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::GR8RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::GR8RegClassID,
      GIM_Try, /*On fail goto*//*Label 792*/ 22058, // Rule ID 619 //
        GIM_CheckConstantInt, /*MI*/0, /*Op*/2, 1,
        // (srl:{ *:[i8] } GR8:{ *:[i8] }:$src1, 1:{ *:[i8] })  =>  (SHR8r1:{ *:[i8] }:{ *:[i32] } GR8:{ *:[i8] }:$src1)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::SHR8r1,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // src1
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 619,
        GIR_Done,
      // Label 792: @22058
      GIM_Try, /*On fail goto*//*Label 793*/ 22088, // Rule ID 615 //
        GIM_RecordInsn, /*DefineMI*/1, /*MI*/0, /*OpIdx*/2, // MIs[1]
        GIM_CheckOpcode, /*MI*/1, TargetOpcode::G_CONSTANT,
        // MIs[1] Operand 1
        // No operand predicates
        GIM_CheckIsSafeToFold, /*InsnID*/1,
        // (srl:{ *:[i8] } GR8:{ *:[i8] }:$src1, (imm:{ *:[i8] }):$src2)  =>  (SHR8ri:{ *:[i8] }:{ *:[i32] } GR8:{ *:[i8] }:$src1, (imm:{ *:[i8] }):$src2)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::SHR8ri,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // src1
        GIR_CopyConstantAsSImm, /*NewInsnID*/0, /*OldInsnID*/1, // src2
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 615,
        GIR_Done,
      // Label 793: @22088
      GIM_Reject,
    // Label 791: @22089
    GIM_Reject,
    // Label 778: @22090
    GIM_Try, /*On fail goto*//*Label 794*/ 22161,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_s16,
      GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_s8,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::GR16RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::GR16RegClassID,
      GIM_Try, /*On fail goto*//*Label 795*/ 22130, // Rule ID 620 //
        GIM_CheckConstantInt, /*MI*/0, /*Op*/2, 1,
        // (srl:{ *:[i16] } GR16:{ *:[i16] }:$src1, 1:{ *:[i8] })  =>  (SHR16r1:{ *:[i16] }:{ *:[i32] } GR16:{ *:[i16] }:$src1)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::SHR16r1,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // src1
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 620,
        GIR_Done,
      // Label 795: @22130
      GIM_Try, /*On fail goto*//*Label 796*/ 22160, // Rule ID 616 //
        GIM_RecordInsn, /*DefineMI*/1, /*MI*/0, /*OpIdx*/2, // MIs[1]
        GIM_CheckOpcode, /*MI*/1, TargetOpcode::G_CONSTANT,
        // MIs[1] Operand 1
        // No operand predicates
        GIM_CheckIsSafeToFold, /*InsnID*/1,
        // (srl:{ *:[i16] } GR16:{ *:[i16] }:$src1, (imm:{ *:[i8] }):$src2)  =>  (SHR16ri:{ *:[i16] }:{ *:[i32] } GR16:{ *:[i16] }:$src1, (imm:{ *:[i8] }):$src2)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::SHR16ri,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // src1
        GIR_CopyConstantAsSImm, /*NewInsnID*/0, /*OldInsnID*/1, // src2
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 616,
        GIR_Done,
      // Label 796: @22160
      GIM_Reject,
    // Label 794: @22161
    GIM_Reject,
    // Label 779: @22162
    GIM_Try, /*On fail goto*//*Label 797*/ 22355,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_s32,
      GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_s8,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::GR32RegClassID,
      GIM_Try, /*On fail goto*//*Label 798*/ 22294, // Rule ID 12215 //
        GIM_CheckFeatures, GIFBS_HasBMI2,
        GIM_RecordInsn, /*DefineMI*/1, /*MI*/0, /*OpIdx*/1, // MIs[1]
        GIM_CheckOpcode, /*MI*/1, TargetOpcode::G_SHL,
        GIM_CheckType, /*MI*/1, /*Op*/1, /*Type*/GILLT_s32,
        GIM_CheckType, /*MI*/1, /*Op*/2, /*Type*/GILLT_s8,
        GIM_CheckRegBankForClass, /*MI*/1, /*Op*/1, /*RC*/X86::GR32RegClassID,
        GIM_RecordInsn, /*DefineMI*/2, /*MI*/1, /*OpIdx*/2, // MIs[2]
        GIM_CheckOpcode, /*MI*/2, TargetOpcode::G_TRUNC,
        GIM_CheckType, /*MI*/2, /*Op*/1, /*Type*/GILLT_s32,
        GIM_RecordInsn, /*DefineMI*/3, /*MI*/2, /*OpIdx*/1, // MIs[3]
        GIM_CheckOpcode, /*MI*/3, TargetOpcode::G_SUB,
        GIM_CheckType, /*MI*/3, /*Op*/1, /*Type*/GILLT_s32,
        GIM_CheckType, /*MI*/3, /*Op*/2, /*Type*/GILLT_s32,
        GIM_CheckConstantInt, /*MI*/3, /*Op*/1, 32,
        GIM_CheckRegBankForClass, /*MI*/3, /*Op*/2, /*RC*/X86::GR32RegClassID,
        GIM_RecordInsn, /*DefineMI*/4, /*MI*/0, /*OpIdx*/2, // MIs[4]
        GIM_CheckOpcode, /*MI*/4, TargetOpcode::G_TRUNC,
        GIM_CheckType, /*MI*/4, /*Op*/1, /*Type*/GILLT_s32,
        GIM_RecordInsn, /*DefineMI*/5, /*MI*/4, /*OpIdx*/1, // MIs[5]
        GIM_CheckOpcode, /*MI*/5, TargetOpcode::G_SUB,
        GIM_CheckType, /*MI*/5, /*Op*/1, /*Type*/GILLT_s32,
        GIM_CheckConstantInt, /*MI*/5, /*Op*/1, 32,
        // MIs[5] lz
        GIM_CheckIsSameOperand, /*MI*/5, /*OpIdx*/2, /*OtherMI*/3, /*OtherOpIdx*/2,
        GIM_CheckIsSafeToFold, /*InsnID*/1,
        GIM_CheckIsSafeToFold, /*InsnID*/2,
        GIM_CheckIsSafeToFold, /*InsnID*/3,
        GIM_CheckIsSafeToFold, /*InsnID*/4,
        GIM_CheckIsSafeToFold, /*InsnID*/5,
        // (srl:{ *:[i32] } (shl:{ *:[i32] } GR32:{ *:[i32] }:$src, (trunc:{ *:[i8] } (sub:{ *:[i32] } 32:{ *:[i32] }, GR32:{ *:[i32] }:$lz))), (trunc:{ *:[i8] } (sub:{ *:[i32] } 32:{ *:[i32] }, GR32:{ *:[i32] }:$lz)))  =>  (BZHI32rr:{ *:[i32] }:{ *:[i32] } GR32:{ *:[i32] }:$src, GR32:{ *:[i32] }:$lz)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::BZHI32rr,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/1, /*OpIdx*/1, // src
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/3, /*OpIdx*/2, // lz
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 12215,
        GIR_Done,
      // Label 798: @22294
      GIM_Try, /*On fail goto*//*Label 799*/ 22320, // Rule ID 621 //
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::GR32RegClassID,
        GIM_CheckConstantInt, /*MI*/0, /*Op*/2, 1,
        // (srl:{ *:[i32] } GR32:{ *:[i32] }:$src1, 1:{ *:[i8] })  =>  (SHR32r1:{ *:[i32] }:{ *:[i32] } GR32:{ *:[i32] }:$src1)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::SHR32r1,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // src1
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 621,
        GIR_Done,
      // Label 799: @22320
      GIM_Try, /*On fail goto*//*Label 800*/ 22354, // Rule ID 617 //
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::GR32RegClassID,
        GIM_RecordInsn, /*DefineMI*/1, /*MI*/0, /*OpIdx*/2, // MIs[1]
        GIM_CheckOpcode, /*MI*/1, TargetOpcode::G_CONSTANT,
        // MIs[1] Operand 1
        // No operand predicates
        GIM_CheckIsSafeToFold, /*InsnID*/1,
        // (srl:{ *:[i32] } GR32:{ *:[i32] }:$src1, (imm:{ *:[i8] }):$src2)  =>  (SHR32ri:{ *:[i32] }:{ *:[i32] } GR32:{ *:[i32] }:$src1, (imm:{ *:[i8] }):$src2)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::SHR32ri,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // src1
        GIR_CopyConstantAsSImm, /*NewInsnID*/0, /*OldInsnID*/1, // src2
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 617,
        GIR_Done,
      // Label 800: @22354
      GIM_Reject,
    // Label 797: @22355
    GIM_Reject,
    // Label 780: @22356
    GIM_Try, /*On fail goto*//*Label 801*/ 22427,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_s64,
      GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_s8,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::GR64RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::GR64RegClassID,
      GIM_Try, /*On fail goto*//*Label 802*/ 22396, // Rule ID 622 //
        GIM_CheckConstantInt, /*MI*/0, /*Op*/2, 1,
        // (srl:{ *:[i64] } GR64:{ *:[i64] }:$src1, 1:{ *:[i8] })  =>  (SHR64r1:{ *:[i64] }:{ *:[i32] } GR64:{ *:[i64] }:$src1)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::SHR64r1,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // src1
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 622,
        GIR_Done,
      // Label 802: @22396
      GIM_Try, /*On fail goto*//*Label 803*/ 22426, // Rule ID 618 //
        GIM_RecordInsn, /*DefineMI*/1, /*MI*/0, /*OpIdx*/2, // MIs[1]
        GIM_CheckOpcode, /*MI*/1, TargetOpcode::G_CONSTANT,
        // MIs[1] Operand 1
        // No operand predicates
        GIM_CheckIsSafeToFold, /*InsnID*/1,
        // (srl:{ *:[i64] } GR64:{ *:[i64] }:$src1, (imm:{ *:[i8] }):$src2)  =>  (SHR64ri:{ *:[i64] }:{ *:[i32] } GR64:{ *:[i64] }:$src1, (imm:{ *:[i8] }):$src2)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::SHR64ri,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // src1
        GIR_CopyConstantAsSImm, /*NewInsnID*/0, /*OldInsnID*/1, // src2
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 618,
        GIR_Done,
      // Label 803: @22426
      GIM_Reject,
    // Label 801: @22427
    GIM_Reject,
    // Label 781: @22428
    GIM_Try, /*On fail goto*//*Label 804*/ 22485,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v2s64,
      GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v2s64,
      GIM_Try, /*On fail goto*//*Label 805*/ 22461, // Rule ID 2731 //
        GIM_CheckFeatures, GIFBS_HasAVX2_NoVLX,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR128RegClassID,
        // (srl:{ *:[v2i64] } VR128:{ *:[v2i64] }:$src1, VR128:{ *:[v2i64] }:$src2)  =>  (VPSRLVQrr:{ *:[v2i64] } VR128:{ *:[v2i64] }:$src1, VR128:{ *:[v2i64] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VPSRLVQrr,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 2731,
        GIR_Done,
      // Label 805: @22461
      GIM_Try, /*On fail goto*//*Label 806*/ 22484, // Rule ID 6541 //
        GIM_CheckFeatures, GIFBS_HasAVX512_HasVLX,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128XRegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR128XRegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR128XRegClassID,
        // (srl:{ *:[v2i64] } VR128X:{ *:[v2i64] }:$src1, VR128X:{ *:[v2i64] }:$src2)  =>  (VPSRLVQZ128rr:{ *:[v2i64] } VR128X:{ *:[v2i64] }:$src1, VR128X:{ *:[v2i64] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VPSRLVQZ128rr,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 6541,
        GIR_Done,
      // Label 806: @22484
      GIM_Reject,
    // Label 804: @22485
    GIM_Reject,
    // Label 782: @22486
    GIM_Try, /*On fail goto*//*Label 807*/ 22543,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v4s32,
      GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v4s32,
      GIM_Try, /*On fail goto*//*Label 808*/ 22519, // Rule ID 2727 //
        GIM_CheckFeatures, GIFBS_HasAVX2_NoVLX,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR128RegClassID,
        // (srl:{ *:[v4i32] } VR128:{ *:[v4i32] }:$src1, VR128:{ *:[v4i32] }:$src2)  =>  (VPSRLVDrr:{ *:[v4i32] } VR128:{ *:[v4i32] }:$src1, VR128:{ *:[v4i32] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VPSRLVDrr,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 2727,
        GIR_Done,
      // Label 808: @22519
      GIM_Try, /*On fail goto*//*Label 809*/ 22542, // Rule ID 6514 //
        GIM_CheckFeatures, GIFBS_HasAVX512_HasVLX,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128XRegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR128XRegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR128XRegClassID,
        // (srl:{ *:[v4i32] } VR128X:{ *:[v4i32] }:$src1, VR128X:{ *:[v4i32] }:$src2)  =>  (VPSRLVDZ128rr:{ *:[v4i32] } VR128X:{ *:[v4i32] }:$src1, VR128X:{ *:[v4i32] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VPSRLVDZ128rr,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 6514,
        GIR_Done,
      // Label 809: @22542
      GIM_Reject,
    // Label 807: @22543
    GIM_Reject,
    // Label 783: @22544
    GIM_Try, /*On fail goto*//*Label 810*/ 22601,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v4s64,
      GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v4s64,
      GIM_Try, /*On fail goto*//*Label 811*/ 22577, // Rule ID 2733 //
        GIM_CheckFeatures, GIFBS_HasAVX2_NoVLX,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR256RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR256RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR256RegClassID,
        // (srl:{ *:[v4i64] } VR256:{ *:[v4i64] }:$src1, VR256:{ *:[v4i64] }:$src2)  =>  (VPSRLVQYrr:{ *:[v4i64] } VR256:{ *:[v4i64] }:$src1, VR256:{ *:[v4i64] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VPSRLVQYrr,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 2733,
        GIR_Done,
      // Label 811: @22577
      GIM_Try, /*On fail goto*//*Label 812*/ 22600, // Rule ID 6532 //
        GIM_CheckFeatures, GIFBS_HasAVX512_HasVLX,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR256XRegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR256XRegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR256XRegClassID,
        // (srl:{ *:[v4i64] } VR256X:{ *:[v4i64] }:$src1, VR256X:{ *:[v4i64] }:$src2)  =>  (VPSRLVQZ256rr:{ *:[v4i64] } VR256X:{ *:[v4i64] }:$src1, VR256X:{ *:[v4i64] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VPSRLVQZ256rr,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 6532,
        GIR_Done,
      // Label 812: @22600
      GIM_Reject,
    // Label 810: @22601
    GIM_Reject,
    // Label 784: @22602
    GIM_Try, /*On fail goto*//*Label 813*/ 22633, // Rule ID 6562 //
      GIM_CheckFeatures, GIFBS_HasBWI_HasVLX,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v8s16,
      GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v8s16,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128XRegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR128XRegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR128XRegClassID,
      // (srl:{ *:[v8i16] } VR128X:{ *:[v8i16] }:$src1, VR128X:{ *:[v8i16] }:$src2)  =>  (VPSRLVWZ128rr:{ *:[v8i16] } VR128X:{ *:[v8i16] }:$src1, VR128X:{ *:[v8i16] }:$src2)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VPSRLVWZ128rr,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 6562,
      GIR_Done,
    // Label 813: @22633
    GIM_Reject,
    // Label 785: @22634
    GIM_Try, /*On fail goto*//*Label 814*/ 22691,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v8s32,
      GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v8s32,
      GIM_Try, /*On fail goto*//*Label 815*/ 22667, // Rule ID 2729 //
        GIM_CheckFeatures, GIFBS_HasAVX2_NoVLX,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR256RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR256RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR256RegClassID,
        // (srl:{ *:[v8i32] } VR256:{ *:[v8i32] }:$src1, VR256:{ *:[v8i32] }:$src2)  =>  (VPSRLVDYrr:{ *:[v8i32] } VR256:{ *:[v8i32] }:$src1, VR256:{ *:[v8i32] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VPSRLVDYrr,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 2729,
        GIR_Done,
      // Label 815: @22667
      GIM_Try, /*On fail goto*//*Label 816*/ 22690, // Rule ID 6505 //
        GIM_CheckFeatures, GIFBS_HasAVX512_HasVLX,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR256XRegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR256XRegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR256XRegClassID,
        // (srl:{ *:[v8i32] } VR256X:{ *:[v8i32] }:$src1, VR256X:{ *:[v8i32] }:$src2)  =>  (VPSRLVDZ256rr:{ *:[v8i32] } VR256X:{ *:[v8i32] }:$src1, VR256X:{ *:[v8i32] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VPSRLVDZ256rr,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 6505,
        GIR_Done,
      // Label 816: @22690
      GIM_Reject,
    // Label 814: @22691
    GIM_Reject,
    // Label 786: @22692
    GIM_Try, /*On fail goto*//*Label 817*/ 22723, // Rule ID 6523 //
      GIM_CheckFeatures, GIFBS_HasAVX512,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v8s64,
      GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v8s64,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR512RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR512RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR512RegClassID,
      // (srl:{ *:[v8i64] } VR512:{ *:[v8i64] }:$src1, VR512:{ *:[v8i64] }:$src2)  =>  (VPSRLVQZrr:{ *:[v8i64] } VR512:{ *:[v8i64] }:$src1, VR512:{ *:[v8i64] }:$src2)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VPSRLVQZrr,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 6523,
      GIR_Done,
    // Label 817: @22723
    GIM_Reject,
    // Label 787: @22724
    GIM_Try, /*On fail goto*//*Label 818*/ 22755, // Rule ID 6556 //
      GIM_CheckFeatures, GIFBS_HasBWI_HasVLX,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v16s16,
      GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v16s16,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR256XRegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR256XRegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR256XRegClassID,
      // (srl:{ *:[v16i16] } VR256X:{ *:[v16i16] }:$src1, VR256X:{ *:[v16i16] }:$src2)  =>  (VPSRLVWZ256rr:{ *:[v16i16] } VR256X:{ *:[v16i16] }:$src1, VR256X:{ *:[v16i16] }:$src2)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VPSRLVWZ256rr,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 6556,
      GIR_Done,
    // Label 818: @22755
    GIM_Reject,
    // Label 788: @22756
    GIM_Try, /*On fail goto*//*Label 819*/ 22787, // Rule ID 6496 //
      GIM_CheckFeatures, GIFBS_HasAVX512,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v16s32,
      GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v16s32,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR512RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR512RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR512RegClassID,
      // (srl:{ *:[v16i32] } VR512:{ *:[v16i32] }:$src1, VR512:{ *:[v16i32] }:$src2)  =>  (VPSRLVDZrr:{ *:[v16i32] } VR512:{ *:[v16i32] }:$src1, VR512:{ *:[v16i32] }:$src2)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VPSRLVDZrr,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 6496,
      GIR_Done,
    // Label 819: @22787
    GIM_Reject,
    // Label 789: @22788
    GIM_Try, /*On fail goto*//*Label 820*/ 22819, // Rule ID 6550 //
      GIM_CheckFeatures, GIFBS_HasBWI,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v32s16,
      GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v32s16,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR512RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR512RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR512RegClassID,
      // (srl:{ *:[v32i16] } VR512:{ *:[v32i16] }:$src1, VR512:{ *:[v32i16] }:$src2)  =>  (VPSRLVWZrr:{ *:[v32i16] } VR512:{ *:[v32i16] }:$src1, VR512:{ *:[v32i16] }:$src2)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VPSRLVWZrr,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 6550,
      GIR_Done,
    // Label 820: @22819
    GIM_Reject,
    // Label 790: @22820
    GIM_Reject,
    // Label 17: @22821
    GIM_SwitchType, /*MI*/0, /*Op*/0, /*[*/1, 23, /*)*//*default:*//*Label 834*/ 23477,
    /*GILLT_s8*//*Label 821*/ 22849,
    /*GILLT_s16*//*Label 822*/ 22921,
    /*GILLT_s32*//*Label 823*/ 22993,
    /*GILLT_s64*//*Label 824*/ 23065, 0, 0, 0,
    /*GILLT_v2s64*//*Label 825*/ 23137, 0,
    /*GILLT_v4s32*//*Label 826*/ 23169,
    /*GILLT_v4s64*//*Label 827*/ 23227, 0,
    /*GILLT_v8s16*//*Label 828*/ 23259,
    /*GILLT_v8s32*//*Label 829*/ 23291,
    /*GILLT_v8s64*//*Label 830*/ 23349, 0, 0,
    /*GILLT_v16s16*//*Label 831*/ 23381,
    /*GILLT_v16s32*//*Label 832*/ 23413, 0, 0,
    /*GILLT_v32s16*//*Label 833*/ 23445,
    // Label 821: @22849
    GIM_Try, /*On fail goto*//*Label 835*/ 22920,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_s8,
      GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_s8,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::GR8RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::GR8RegClassID,
      GIM_Try, /*On fail goto*//*Label 836*/ 22889, // Rule ID 643 //
        GIM_CheckConstantInt, /*MI*/0, /*Op*/2, 1,
        // (sra:{ *:[i8] } GR8:{ *:[i8] }:$src1, 1:{ *:[i8] })  =>  (SAR8r1:{ *:[i8] }:{ *:[i32] } GR8:{ *:[i8] }:$src1)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::SAR8r1,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // src1
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 643,
        GIR_Done,
      // Label 836: @22889
      GIM_Try, /*On fail goto*//*Label 837*/ 22919, // Rule ID 639 //
        GIM_RecordInsn, /*DefineMI*/1, /*MI*/0, /*OpIdx*/2, // MIs[1]
        GIM_CheckOpcode, /*MI*/1, TargetOpcode::G_CONSTANT,
        // MIs[1] Operand 1
        // No operand predicates
        GIM_CheckIsSafeToFold, /*InsnID*/1,
        // (sra:{ *:[i8] } GR8:{ *:[i8] }:$src1, (imm:{ *:[i8] }):$src2)  =>  (SAR8ri:{ *:[i8] }:{ *:[i32] } GR8:{ *:[i8] }:$src1, (imm:{ *:[i8] }):$src2)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::SAR8ri,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // src1
        GIR_CopyConstantAsSImm, /*NewInsnID*/0, /*OldInsnID*/1, // src2
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 639,
        GIR_Done,
      // Label 837: @22919
      GIM_Reject,
    // Label 835: @22920
    GIM_Reject,
    // Label 822: @22921
    GIM_Try, /*On fail goto*//*Label 838*/ 22992,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_s16,
      GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_s8,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::GR16RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::GR16RegClassID,
      GIM_Try, /*On fail goto*//*Label 839*/ 22961, // Rule ID 644 //
        GIM_CheckConstantInt, /*MI*/0, /*Op*/2, 1,
        // (sra:{ *:[i16] } GR16:{ *:[i16] }:$src1, 1:{ *:[i8] })  =>  (SAR16r1:{ *:[i16] }:{ *:[i32] } GR16:{ *:[i16] }:$src1)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::SAR16r1,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // src1
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 644,
        GIR_Done,
      // Label 839: @22961
      GIM_Try, /*On fail goto*//*Label 840*/ 22991, // Rule ID 640 //
        GIM_RecordInsn, /*DefineMI*/1, /*MI*/0, /*OpIdx*/2, // MIs[1]
        GIM_CheckOpcode, /*MI*/1, TargetOpcode::G_CONSTANT,
        // MIs[1] Operand 1
        // No operand predicates
        GIM_CheckIsSafeToFold, /*InsnID*/1,
        // (sra:{ *:[i16] } GR16:{ *:[i16] }:$src1, (imm:{ *:[i8] }):$src2)  =>  (SAR16ri:{ *:[i16] }:{ *:[i32] } GR16:{ *:[i16] }:$src1, (imm:{ *:[i8] }):$src2)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::SAR16ri,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // src1
        GIR_CopyConstantAsSImm, /*NewInsnID*/0, /*OldInsnID*/1, // src2
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 640,
        GIR_Done,
      // Label 840: @22991
      GIM_Reject,
    // Label 838: @22992
    GIM_Reject,
    // Label 823: @22993
    GIM_Try, /*On fail goto*//*Label 841*/ 23064,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_s32,
      GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_s8,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::GR32RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::GR32RegClassID,
      GIM_Try, /*On fail goto*//*Label 842*/ 23033, // Rule ID 645 //
        GIM_CheckConstantInt, /*MI*/0, /*Op*/2, 1,
        // (sra:{ *:[i32] } GR32:{ *:[i32] }:$src1, 1:{ *:[i8] })  =>  (SAR32r1:{ *:[i32] }:{ *:[i32] } GR32:{ *:[i32] }:$src1)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::SAR32r1,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // src1
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 645,
        GIR_Done,
      // Label 842: @23033
      GIM_Try, /*On fail goto*//*Label 843*/ 23063, // Rule ID 641 //
        GIM_RecordInsn, /*DefineMI*/1, /*MI*/0, /*OpIdx*/2, // MIs[1]
        GIM_CheckOpcode, /*MI*/1, TargetOpcode::G_CONSTANT,
        // MIs[1] Operand 1
        // No operand predicates
        GIM_CheckIsSafeToFold, /*InsnID*/1,
        // (sra:{ *:[i32] } GR32:{ *:[i32] }:$src1, (imm:{ *:[i8] }):$src2)  =>  (SAR32ri:{ *:[i32] }:{ *:[i32] } GR32:{ *:[i32] }:$src1, (imm:{ *:[i8] }):$src2)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::SAR32ri,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // src1
        GIR_CopyConstantAsSImm, /*NewInsnID*/0, /*OldInsnID*/1, // src2
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 641,
        GIR_Done,
      // Label 843: @23063
      GIM_Reject,
    // Label 841: @23064
    GIM_Reject,
    // Label 824: @23065
    GIM_Try, /*On fail goto*//*Label 844*/ 23136,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_s64,
      GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_s8,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::GR64RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::GR64RegClassID,
      GIM_Try, /*On fail goto*//*Label 845*/ 23105, // Rule ID 646 //
        GIM_CheckConstantInt, /*MI*/0, /*Op*/2, 1,
        // (sra:{ *:[i64] } GR64:{ *:[i64] }:$src1, 1:{ *:[i8] })  =>  (SAR64r1:{ *:[i64] }:{ *:[i32] } GR64:{ *:[i64] }:$src1)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::SAR64r1,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // src1
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 646,
        GIR_Done,
      // Label 845: @23105
      GIM_Try, /*On fail goto*//*Label 846*/ 23135, // Rule ID 642 //
        GIM_RecordInsn, /*DefineMI*/1, /*MI*/0, /*OpIdx*/2, // MIs[1]
        GIM_CheckOpcode, /*MI*/1, TargetOpcode::G_CONSTANT,
        // MIs[1] Operand 1
        // No operand predicates
        GIM_CheckIsSafeToFold, /*InsnID*/1,
        // (sra:{ *:[i64] } GR64:{ *:[i64] }:$src1, (imm:{ *:[i8] }):$src2)  =>  (SAR64ri:{ *:[i64] }:{ *:[i32] } GR64:{ *:[i64] }:$src1, (imm:{ *:[i8] }):$src2)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::SAR64ri,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // src1
        GIR_CopyConstantAsSImm, /*NewInsnID*/0, /*OldInsnID*/1, // src2
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 642,
        GIR_Done,
      // Label 846: @23135
      GIM_Reject,
    // Label 844: @23136
    GIM_Reject,
    // Label 825: @23137
    GIM_Try, /*On fail goto*//*Label 847*/ 23168, // Rule ID 6469 //
      GIM_CheckFeatures, GIFBS_HasAVX512_HasVLX,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v2s64,
      GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v2s64,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128XRegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR128XRegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR128XRegClassID,
      // (sra:{ *:[v2i64] } VR128X:{ *:[v2i64] }:$src1, VR128X:{ *:[v2i64] }:$src2)  =>  (VPSRAVQZ128rr:{ *:[v2i64] } VR128X:{ *:[v2i64] }:$src1, VR128X:{ *:[v2i64] }:$src2)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VPSRAVQZ128rr,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 6469,
      GIR_Done,
    // Label 847: @23168
    GIM_Reject,
    // Label 826: @23169
    GIM_Try, /*On fail goto*//*Label 848*/ 23226,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v4s32,
      GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v4s32,
      GIM_Try, /*On fail goto*//*Label 849*/ 23202, // Rule ID 2735 //
        GIM_CheckFeatures, GIFBS_HasAVX2_NoVLX,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR128RegClassID,
        // (sra:{ *:[v4i32] } VR128:{ *:[v4i32] }:$src1, VR128:{ *:[v4i32] }:$src2)  =>  (VPSRAVDrr:{ *:[v4i32] } VR128:{ *:[v4i32] }:$src1, VR128:{ *:[v4i32] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VPSRAVDrr,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 2735,
        GIR_Done,
      // Label 849: @23202
      GIM_Try, /*On fail goto*//*Label 850*/ 23225, // Rule ID 6442 //
        GIM_CheckFeatures, GIFBS_HasAVX512_HasVLX,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128XRegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR128XRegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR128XRegClassID,
        // (sra:{ *:[v4i32] } VR128X:{ *:[v4i32] }:$src1, VR128X:{ *:[v4i32] }:$src2)  =>  (VPSRAVDZ128rr:{ *:[v4i32] } VR128X:{ *:[v4i32] }:$src1, VR128X:{ *:[v4i32] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VPSRAVDZ128rr,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 6442,
        GIR_Done,
      // Label 850: @23225
      GIM_Reject,
    // Label 848: @23226
    GIM_Reject,
    // Label 827: @23227
    GIM_Try, /*On fail goto*//*Label 851*/ 23258, // Rule ID 6460 //
      GIM_CheckFeatures, GIFBS_HasAVX512_HasVLX,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v4s64,
      GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v4s64,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR256XRegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR256XRegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR256XRegClassID,
      // (sra:{ *:[v4i64] } VR256X:{ *:[v4i64] }:$src1, VR256X:{ *:[v4i64] }:$src2)  =>  (VPSRAVQZ256rr:{ *:[v4i64] } VR256X:{ *:[v4i64] }:$src1, VR256X:{ *:[v4i64] }:$src2)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VPSRAVQZ256rr,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 6460,
      GIR_Done,
    // Label 851: @23258
    GIM_Reject,
    // Label 828: @23259
    GIM_Try, /*On fail goto*//*Label 852*/ 23290, // Rule ID 6490 //
      GIM_CheckFeatures, GIFBS_HasBWI_HasVLX,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v8s16,
      GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v8s16,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128XRegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR128XRegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR128XRegClassID,
      // (sra:{ *:[v8i16] } VR128X:{ *:[v8i16] }:$src1, VR128X:{ *:[v8i16] }:$src2)  =>  (VPSRAVWZ128rr:{ *:[v8i16] } VR128X:{ *:[v8i16] }:$src1, VR128X:{ *:[v8i16] }:$src2)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VPSRAVWZ128rr,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 6490,
      GIR_Done,
    // Label 852: @23290
    GIM_Reject,
    // Label 829: @23291
    GIM_Try, /*On fail goto*//*Label 853*/ 23348,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v8s32,
      GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v8s32,
      GIM_Try, /*On fail goto*//*Label 854*/ 23324, // Rule ID 2737 //
        GIM_CheckFeatures, GIFBS_HasAVX2_NoVLX,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR256RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR256RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR256RegClassID,
        // (sra:{ *:[v8i32] } VR256:{ *:[v8i32] }:$src1, VR256:{ *:[v8i32] }:$src2)  =>  (VPSRAVDYrr:{ *:[v8i32] } VR256:{ *:[v8i32] }:$src1, VR256:{ *:[v8i32] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VPSRAVDYrr,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 2737,
        GIR_Done,
      // Label 854: @23324
      GIM_Try, /*On fail goto*//*Label 855*/ 23347, // Rule ID 6433 //
        GIM_CheckFeatures, GIFBS_HasAVX512_HasVLX,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR256XRegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR256XRegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR256XRegClassID,
        // (sra:{ *:[v8i32] } VR256X:{ *:[v8i32] }:$src1, VR256X:{ *:[v8i32] }:$src2)  =>  (VPSRAVDZ256rr:{ *:[v8i32] } VR256X:{ *:[v8i32] }:$src1, VR256X:{ *:[v8i32] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VPSRAVDZ256rr,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 6433,
        GIR_Done,
      // Label 855: @23347
      GIM_Reject,
    // Label 853: @23348
    GIM_Reject,
    // Label 830: @23349
    GIM_Try, /*On fail goto*//*Label 856*/ 23380, // Rule ID 6451 //
      GIM_CheckFeatures, GIFBS_HasAVX512,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v8s64,
      GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v8s64,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR512RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR512RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR512RegClassID,
      // (sra:{ *:[v8i64] } VR512:{ *:[v8i64] }:$src1, VR512:{ *:[v8i64] }:$src2)  =>  (VPSRAVQZrr:{ *:[v8i64] } VR512:{ *:[v8i64] }:$src1, VR512:{ *:[v8i64] }:$src2)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VPSRAVQZrr,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 6451,
      GIR_Done,
    // Label 856: @23380
    GIM_Reject,
    // Label 831: @23381
    GIM_Try, /*On fail goto*//*Label 857*/ 23412, // Rule ID 6484 //
      GIM_CheckFeatures, GIFBS_HasBWI_HasVLX,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v16s16,
      GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v16s16,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR256XRegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR256XRegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR256XRegClassID,
      // (sra:{ *:[v16i16] } VR256X:{ *:[v16i16] }:$src1, VR256X:{ *:[v16i16] }:$src2)  =>  (VPSRAVWZ256rr:{ *:[v16i16] } VR256X:{ *:[v16i16] }:$src1, VR256X:{ *:[v16i16] }:$src2)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VPSRAVWZ256rr,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 6484,
      GIR_Done,
    // Label 857: @23412
    GIM_Reject,
    // Label 832: @23413
    GIM_Try, /*On fail goto*//*Label 858*/ 23444, // Rule ID 6424 //
      GIM_CheckFeatures, GIFBS_HasAVX512,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v16s32,
      GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v16s32,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR512RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR512RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR512RegClassID,
      // (sra:{ *:[v16i32] } VR512:{ *:[v16i32] }:$src1, VR512:{ *:[v16i32] }:$src2)  =>  (VPSRAVDZrr:{ *:[v16i32] } VR512:{ *:[v16i32] }:$src1, VR512:{ *:[v16i32] }:$src2)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VPSRAVDZrr,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 6424,
      GIR_Done,
    // Label 858: @23444
    GIM_Reject,
    // Label 833: @23445
    GIM_Try, /*On fail goto*//*Label 859*/ 23476, // Rule ID 6478 //
      GIM_CheckFeatures, GIFBS_HasBWI,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v32s16,
      GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v32s16,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR512RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR512RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR512RegClassID,
      // (sra:{ *:[v32i16] } VR512:{ *:[v32i16] }:$src1, VR512:{ *:[v32i16] }:$src2)  =>  (VPSRAVWZrr:{ *:[v32i16] } VR512:{ *:[v32i16] }:$src1, VR512:{ *:[v32i16] }:$src2)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VPSRAVWZrr,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 6478,
      GIR_Done,
    // Label 859: @23476
    GIM_Reject,
    // Label 834: @23477
    GIM_Reject,
    // Label 18: @23478
    GIM_SwitchType, /*MI*/0, /*Op*/0, /*[*/3, 20, /*)*//*default:*//*Label 869*/ 24090,
    /*GILLT_s32*//*Label 860*/ 23501,
    /*GILLT_s64*//*Label 861*/ 23608,
    /*GILLT_s80*//*Label 862*/ 23715, 0, 0,
    /*GILLT_v2s64*//*Label 863*/ 23748, 0,
    /*GILLT_v4s32*//*Label 864*/ 23829,
    /*GILLT_v4s64*//*Label 865*/ 23910, 0, 0,
    /*GILLT_v8s32*//*Label 866*/ 23968,
    /*GILLT_v8s64*//*Label 867*/ 24026, 0, 0, 0,
    /*GILLT_v16s32*//*Label 868*/ 24058,
    // Label 860: @23501
    GIM_Try, /*On fail goto*//*Label 870*/ 23607,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_s32,
      GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_s32,
      GIM_Try, /*On fail goto*//*Label 871*/ 23537, // Rule ID 740 //
        GIM_CheckFeatures, GIFBS_FPStackf32,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::RFP32RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::RFP32RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::RFP32RegClassID,
        // (fadd:{ *:[f32] } RFP32:{ *:[f32] }:$src1, RFP32:{ *:[f32] }:$src2)  =>  (ADD_Fp32:{ *:[f32] }:{ *:[i16] } RFP32:{ *:[f32] }:$src1, RFP32:{ *:[f32] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::ADD_Fp32,
        GIR_AddImplicitDef, /*InsnID*/0, X86::FPSW,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 740,
        GIR_Done,
      // Label 871: @23537
      GIM_Try, /*On fail goto*//*Label 872*/ 23560, // Rule ID 1577 //
        GIM_CheckFeatures, GIFBS_UseAVX,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::FR32RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::FR32RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::FR32RegClassID,
        // (fadd:{ *:[f32] } FR32:{ *:[f32] }:$src1, FR32:{ *:[f32] }:$src2)  =>  (VADDSSrr:{ *:[f32] } FR32:{ *:[f32] }:$src1, FR32:{ *:[f32] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VADDSSrr,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 1577,
        GIR_Done,
      // Label 872: @23560
      GIM_Try, /*On fail goto*//*Label 873*/ 23583, // Rule ID 1581 //
        GIM_CheckFeatures, GIFBS_UseSSE1,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::FR32RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::FR32RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::FR32RegClassID,
        // (fadd:{ *:[f32] } FR32:{ *:[f32] }:$src1, FR32:{ *:[f32] }:$src2)  =>  (ADDSSrr:{ *:[f32] } FR32:{ *:[f32] }:$src1, FR32:{ *:[f32] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::ADDSSrr,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 1581,
        GIR_Done,
      // Label 873: @23583
      GIM_Try, /*On fail goto*//*Label 874*/ 23606, // Rule ID 5066 //
        GIM_CheckFeatures, GIFBS_HasAVX512,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::FR32XRegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::FR32XRegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::FR32XRegClassID,
        // (fadd:{ *:[f32] } FR32X:{ *:[f32] }:$src1, FR32X:{ *:[f32] }:$src2)  =>  (VADDSSZrr:{ *:[f32] } FR32X:{ *:[f32] }:$src1, FR32X:{ *:[f32] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VADDSSZrr,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 5066,
        GIR_Done,
      // Label 874: @23606
      GIM_Reject,
    // Label 870: @23607
    GIM_Reject,
    // Label 861: @23608
    GIM_Try, /*On fail goto*//*Label 875*/ 23714,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_s64,
      GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_s64,
      GIM_Try, /*On fail goto*//*Label 876*/ 23644, // Rule ID 741 //
        GIM_CheckFeatures, GIFBS_FPStackf64,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::RFP64RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::RFP64RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::RFP64RegClassID,
        // (fadd:{ *:[f64] } RFP64:{ *:[f64] }:$src1, RFP64:{ *:[f64] }:$src2)  =>  (ADD_Fp64:{ *:[f64] }:{ *:[i16] } RFP64:{ *:[f64] }:$src1, RFP64:{ *:[f64] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::ADD_Fp64,
        GIR_AddImplicitDef, /*InsnID*/0, X86::FPSW,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 741,
        GIR_Done,
      // Label 876: @23644
      GIM_Try, /*On fail goto*//*Label 877*/ 23667, // Rule ID 1579 //
        GIM_CheckFeatures, GIFBS_UseAVX,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::FR64RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::FR64RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::FR64RegClassID,
        // (fadd:{ *:[f64] } FR64:{ *:[f64] }:$src1, FR64:{ *:[f64] }:$src2)  =>  (VADDSDrr:{ *:[f64] } FR64:{ *:[f64] }:$src1, FR64:{ *:[f64] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VADDSDrr,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 1579,
        GIR_Done,
      // Label 877: @23667
      GIM_Try, /*On fail goto*//*Label 878*/ 23690, // Rule ID 1583 //
        GIM_CheckFeatures, GIFBS_UseSSE2,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::FR64RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::FR64RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::FR64RegClassID,
        // (fadd:{ *:[f64] } FR64:{ *:[f64] }:$src1, FR64:{ *:[f64] }:$src2)  =>  (ADDSDrr:{ *:[f64] } FR64:{ *:[f64] }:$src1, FR64:{ *:[f64] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::ADDSDrr,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 1583,
        GIR_Done,
      // Label 878: @23690
      GIM_Try, /*On fail goto*//*Label 879*/ 23713, // Rule ID 5077 //
        GIM_CheckFeatures, GIFBS_HasAVX512,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::FR64XRegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::FR64XRegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::FR64XRegClassID,
        // (fadd:{ *:[f64] } FR64X:{ *:[f64] }:$src1, FR64X:{ *:[f64] }:$src2)  =>  (VADDSDZrr:{ *:[f64] } FR64X:{ *:[f64] }:$src1, FR64X:{ *:[f64] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VADDSDZrr,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 5077,
        GIR_Done,
      // Label 879: @23713
      GIM_Reject,
    // Label 875: @23714
    GIM_Reject,
    // Label 862: @23715
    GIM_Try, /*On fail goto*//*Label 880*/ 23747, // Rule ID 742 //
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_s80,
      GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_s80,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::RFP80RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::RFP80RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::RFP80RegClassID,
      // (fadd:{ *:[f80] } RFP80:{ *:[f80] }:$src1, RFP80:{ *:[f80] }:$src2)  =>  (ADD_Fp80:{ *:[f80] }:{ *:[i16] } RFP80:{ *:[f80] }:$src1, RFP80:{ *:[f80] }:$src2)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::ADD_Fp80,
      GIR_AddImplicitDef, /*InsnID*/0, X86::FPSW,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 742,
      GIR_Done,
    // Label 880: @23747
    GIM_Reject,
    // Label 863: @23748
    GIM_Try, /*On fail goto*//*Label 881*/ 23828,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v2s64,
      GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v2s64,
      GIM_Try, /*On fail goto*//*Label 882*/ 23781, // Rule ID 1567 //
        GIM_CheckFeatures, GIFBS_HasAVX_NoVLX,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR128RegClassID,
        // (fadd:{ *:[v2f64] } VR128:{ *:[v2f64] }:$src1, VR128:{ *:[v2f64] }:$src2)  =>  (VADDPDrr:{ *:[v2f64] } VR128:{ *:[v2f64] }:$src1, VR128:{ *:[v2f64] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VADDPDrr,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 1567,
        GIR_Done,
      // Label 882: @23781
      GIM_Try, /*On fail goto*//*Label 883*/ 23804, // Rule ID 1575 //
        GIM_CheckFeatures, GIFBS_UseSSE2,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR128RegClassID,
        // (fadd:{ *:[v2f64] } VR128:{ *:[v2f64] }:$src1, VR128:{ *:[v2f64] }:$src2)  =>  (ADDPDrr:{ *:[v2f64] } VR128:{ *:[v2f64] }:$src1, VR128:{ *:[v2f64] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::ADDPDrr,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 1575,
        GIR_Done,
      // Label 883: @23804
      GIM_Try, /*On fail goto*//*Label 884*/ 23827, // Rule ID 5236 //
        GIM_CheckFeatures, GIFBS_HasAVX512_HasVLX,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128XRegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR128XRegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR128XRegClassID,
        // (fadd:{ *:[v2f64] } VR128X:{ *:[v2f64] }:$src1, VR128X:{ *:[v2f64] }:$src2)  =>  (VADDPDZ128rr:{ *:[v2f64] } VR128X:{ *:[v2f64] }:$src1, VR128X:{ *:[v2f64] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VADDPDZ128rr,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 5236,
        GIR_Done,
      // Label 884: @23827
      GIM_Reject,
    // Label 881: @23828
    GIM_Reject,
    // Label 864: @23829
    GIM_Try, /*On fail goto*//*Label 885*/ 23909,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v4s32,
      GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v4s32,
      GIM_Try, /*On fail goto*//*Label 886*/ 23862, // Rule ID 1565 //
        GIM_CheckFeatures, GIFBS_HasAVX_NoVLX,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR128RegClassID,
        // (fadd:{ *:[v4f32] } VR128:{ *:[v4f32] }:$src1, VR128:{ *:[v4f32] }:$src2)  =>  (VADDPSrr:{ *:[v4f32] } VR128:{ *:[v4f32] }:$src1, VR128:{ *:[v4f32] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VADDPSrr,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 1565,
        GIR_Done,
      // Label 886: @23862
      GIM_Try, /*On fail goto*//*Label 887*/ 23885, // Rule ID 1573 //
        GIM_CheckFeatures, GIFBS_UseSSE1,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR128RegClassID,
        // (fadd:{ *:[v4f32] } VR128:{ *:[v4f32] }:$src1, VR128:{ *:[v4f32] }:$src2)  =>  (ADDPSrr:{ *:[v4f32] } VR128:{ *:[v4f32] }:$src1, VR128:{ *:[v4f32] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::ADDPSrr,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 1573,
        GIR_Done,
      // Label 887: @23885
      GIM_Try, /*On fail goto*//*Label 888*/ 23908, // Rule ID 5218 //
        GIM_CheckFeatures, GIFBS_HasAVX512_HasVLX,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128XRegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR128XRegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR128XRegClassID,
        // (fadd:{ *:[v4f32] } VR128X:{ *:[v4f32] }:$src1, VR128X:{ *:[v4f32] }:$src2)  =>  (VADDPSZ128rr:{ *:[v4f32] } VR128X:{ *:[v4f32] }:$src1, VR128X:{ *:[v4f32] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VADDPSZ128rr,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 5218,
        GIR_Done,
      // Label 888: @23908
      GIM_Reject,
    // Label 885: @23909
    GIM_Reject,
    // Label 865: @23910
    GIM_Try, /*On fail goto*//*Label 889*/ 23967,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v4s64,
      GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v4s64,
      GIM_Try, /*On fail goto*//*Label 890*/ 23943, // Rule ID 1571 //
        GIM_CheckFeatures, GIFBS_HasAVX_NoVLX,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR256RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR256RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR256RegClassID,
        // (fadd:{ *:[v4f64] } VR256:{ *:[v4f64] }:$src1, VR256:{ *:[v4f64] }:$src2)  =>  (VADDPDYrr:{ *:[v4f64] } VR256:{ *:[v4f64] }:$src1, VR256:{ *:[v4f64] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VADDPDYrr,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 1571,
        GIR_Done,
      // Label 890: @23943
      GIM_Try, /*On fail goto*//*Label 891*/ 23966, // Rule ID 5245 //
        GIM_CheckFeatures, GIFBS_HasAVX512_HasVLX,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR256XRegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR256XRegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR256XRegClassID,
        // (fadd:{ *:[v4f64] } VR256X:{ *:[v4f64] }:$src1, VR256X:{ *:[v4f64] }:$src2)  =>  (VADDPDZ256rr:{ *:[v4f64] } VR256X:{ *:[v4f64] }:$src1, VR256X:{ *:[v4f64] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VADDPDZ256rr,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 5245,
        GIR_Done,
      // Label 891: @23966
      GIM_Reject,
    // Label 889: @23967
    GIM_Reject,
    // Label 866: @23968
    GIM_Try, /*On fail goto*//*Label 892*/ 24025,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v8s32,
      GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v8s32,
      GIM_Try, /*On fail goto*//*Label 893*/ 24001, // Rule ID 1569 //
        GIM_CheckFeatures, GIFBS_HasAVX_NoVLX,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR256RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR256RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR256RegClassID,
        // (fadd:{ *:[v8f32] } VR256:{ *:[v8f32] }:$src1, VR256:{ *:[v8f32] }:$src2)  =>  (VADDPSYrr:{ *:[v8f32] } VR256:{ *:[v8f32] }:$src1, VR256:{ *:[v8f32] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VADDPSYrr,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 1569,
        GIR_Done,
      // Label 893: @24001
      GIM_Try, /*On fail goto*//*Label 894*/ 24024, // Rule ID 5227 //
        GIM_CheckFeatures, GIFBS_HasAVX512_HasVLX,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR256XRegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR256XRegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR256XRegClassID,
        // (fadd:{ *:[v8f32] } VR256X:{ *:[v8f32] }:$src1, VR256X:{ *:[v8f32] }:$src2)  =>  (VADDPSZ256rr:{ *:[v8f32] } VR256X:{ *:[v8f32] }:$src1, VR256X:{ *:[v8f32] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VADDPSZ256rr,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 5227,
        GIR_Done,
      // Label 894: @24024
      GIM_Reject,
    // Label 892: @24025
    GIM_Reject,
    // Label 867: @24026
    GIM_Try, /*On fail goto*//*Label 895*/ 24057, // Rule ID 5209 //
      GIM_CheckFeatures, GIFBS_HasAVX512,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v8s64,
      GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v8s64,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR512RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR512RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR512RegClassID,
      // (fadd:{ *:[v8f64] } VR512:{ *:[v8f64] }:$src1, VR512:{ *:[v8f64] }:$src2)  =>  (VADDPDZrr:{ *:[v8f64] } VR512:{ *:[v8f64] }:$src1, VR512:{ *:[v8f64] }:$src2)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VADDPDZrr,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 5209,
      GIR_Done,
    // Label 895: @24057
    GIM_Reject,
    // Label 868: @24058
    GIM_Try, /*On fail goto*//*Label 896*/ 24089, // Rule ID 5200 //
      GIM_CheckFeatures, GIFBS_HasAVX512,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v16s32,
      GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v16s32,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR512RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR512RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR512RegClassID,
      // (fadd:{ *:[v16f32] } VR512:{ *:[v16f32] }:$src1, VR512:{ *:[v16f32] }:$src2)  =>  (VADDPSZrr:{ *:[v16f32] } VR512:{ *:[v16f32] }:$src1, VR512:{ *:[v16f32] }:$src2)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VADDPSZrr,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 5200,
      GIR_Done,
    // Label 896: @24089
    GIM_Reject,
    // Label 869: @24090
    GIM_Reject,
    // Label 19: @24091
    GIM_SwitchType, /*MI*/0, /*Op*/0, /*[*/3, 20, /*)*//*default:*//*Label 906*/ 24703,
    /*GILLT_s32*//*Label 897*/ 24114,
    /*GILLT_s64*//*Label 898*/ 24221,
    /*GILLT_s80*//*Label 899*/ 24328, 0, 0,
    /*GILLT_v2s64*//*Label 900*/ 24361, 0,
    /*GILLT_v4s32*//*Label 901*/ 24442,
    /*GILLT_v4s64*//*Label 902*/ 24523, 0, 0,
    /*GILLT_v8s32*//*Label 903*/ 24581,
    /*GILLT_v8s64*//*Label 904*/ 24639, 0, 0, 0,
    /*GILLT_v16s32*//*Label 905*/ 24671,
    // Label 897: @24114
    GIM_Try, /*On fail goto*//*Label 907*/ 24220,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_s32,
      GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_s32,
      GIM_Try, /*On fail goto*//*Label 908*/ 24150, // Rule ID 743 //
        GIM_CheckFeatures, GIFBS_FPStackf32,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::RFP32RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::RFP32RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::RFP32RegClassID,
        // (fsub:{ *:[f32] } RFP32:{ *:[f32] }:$src1, RFP32:{ *:[f32] }:$src2)  =>  (SUB_Fp32:{ *:[f32] }:{ *:[i16] } RFP32:{ *:[f32] }:$src1, RFP32:{ *:[f32] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::SUB_Fp32,
        GIR_AddImplicitDef, /*InsnID*/0, X86::FPSW,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 743,
        GIR_Done,
      // Label 908: @24150
      GIM_Try, /*On fail goto*//*Label 909*/ 24173, // Rule ID 1617 //
        GIM_CheckFeatures, GIFBS_UseAVX,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::FR32RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::FR32RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::FR32RegClassID,
        // (fsub:{ *:[f32] } FR32:{ *:[f32] }:$src1, FR32:{ *:[f32] }:$src2)  =>  (VSUBSSrr:{ *:[f32] } FR32:{ *:[f32] }:$src1, FR32:{ *:[f32] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VSUBSSrr,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 1617,
        GIR_Done,
      // Label 909: @24173
      GIM_Try, /*On fail goto*//*Label 910*/ 24196, // Rule ID 1621 //
        GIM_CheckFeatures, GIFBS_UseSSE1,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::FR32RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::FR32RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::FR32RegClassID,
        // (fsub:{ *:[f32] } FR32:{ *:[f32] }:$src1, FR32:{ *:[f32] }:$src2)  =>  (SUBSSrr:{ *:[f32] } FR32:{ *:[f32] }:$src1, FR32:{ *:[f32] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::SUBSSrr,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 1621,
        GIR_Done,
      // Label 910: @24196
      GIM_Try, /*On fail goto*//*Label 911*/ 24219, // Rule ID 5110 //
        GIM_CheckFeatures, GIFBS_HasAVX512,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::FR32XRegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::FR32XRegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::FR32XRegClassID,
        // (fsub:{ *:[f32] } FR32X:{ *:[f32] }:$src1, FR32X:{ *:[f32] }:$src2)  =>  (VSUBSSZrr:{ *:[f32] } FR32X:{ *:[f32] }:$src1, FR32X:{ *:[f32] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VSUBSSZrr,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 5110,
        GIR_Done,
      // Label 911: @24219
      GIM_Reject,
    // Label 907: @24220
    GIM_Reject,
    // Label 898: @24221
    GIM_Try, /*On fail goto*//*Label 912*/ 24327,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_s64,
      GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_s64,
      GIM_Try, /*On fail goto*//*Label 913*/ 24257, // Rule ID 744 //
        GIM_CheckFeatures, GIFBS_FPStackf64,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::RFP64RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::RFP64RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::RFP64RegClassID,
        // (fsub:{ *:[f64] } RFP64:{ *:[f64] }:$src1, RFP64:{ *:[f64] }:$src2)  =>  (SUB_Fp64:{ *:[f64] }:{ *:[i16] } RFP64:{ *:[f64] }:$src1, RFP64:{ *:[f64] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::SUB_Fp64,
        GIR_AddImplicitDef, /*InsnID*/0, X86::FPSW,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 744,
        GIR_Done,
      // Label 913: @24257
      GIM_Try, /*On fail goto*//*Label 914*/ 24280, // Rule ID 1619 //
        GIM_CheckFeatures, GIFBS_UseAVX,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::FR64RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::FR64RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::FR64RegClassID,
        // (fsub:{ *:[f64] } FR64:{ *:[f64] }:$src1, FR64:{ *:[f64] }:$src2)  =>  (VSUBSDrr:{ *:[f64] } FR64:{ *:[f64] }:$src1, FR64:{ *:[f64] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VSUBSDrr,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 1619,
        GIR_Done,
      // Label 914: @24280
      GIM_Try, /*On fail goto*//*Label 915*/ 24303, // Rule ID 1623 //
        GIM_CheckFeatures, GIFBS_UseSSE2,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::FR64RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::FR64RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::FR64RegClassID,
        // (fsub:{ *:[f64] } FR64:{ *:[f64] }:$src1, FR64:{ *:[f64] }:$src2)  =>  (SUBSDrr:{ *:[f64] } FR64:{ *:[f64] }:$src1, FR64:{ *:[f64] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::SUBSDrr,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 1623,
        GIR_Done,
      // Label 915: @24303
      GIM_Try, /*On fail goto*//*Label 916*/ 24326, // Rule ID 5121 //
        GIM_CheckFeatures, GIFBS_HasAVX512,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::FR64XRegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::FR64XRegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::FR64XRegClassID,
        // (fsub:{ *:[f64] } FR64X:{ *:[f64] }:$src1, FR64X:{ *:[f64] }:$src2)  =>  (VSUBSDZrr:{ *:[f64] } FR64X:{ *:[f64] }:$src1, FR64X:{ *:[f64] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VSUBSDZrr,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 5121,
        GIR_Done,
      // Label 916: @24326
      GIM_Reject,
    // Label 912: @24327
    GIM_Reject,
    // Label 899: @24328
    GIM_Try, /*On fail goto*//*Label 917*/ 24360, // Rule ID 745 //
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_s80,
      GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_s80,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::RFP80RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::RFP80RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::RFP80RegClassID,
      // (fsub:{ *:[f80] } RFP80:{ *:[f80] }:$src1, RFP80:{ *:[f80] }:$src2)  =>  (SUB_Fp80:{ *:[f80] }:{ *:[i16] } RFP80:{ *:[f80] }:$src1, RFP80:{ *:[f80] }:$src2)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::SUB_Fp80,
      GIR_AddImplicitDef, /*InsnID*/0, X86::FPSW,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 745,
      GIR_Done,
    // Label 917: @24360
    GIM_Reject,
    // Label 900: @24361
    GIM_Try, /*On fail goto*//*Label 918*/ 24441,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v2s64,
      GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v2s64,
      GIM_Try, /*On fail goto*//*Label 919*/ 24394, // Rule ID 1607 //
        GIM_CheckFeatures, GIFBS_HasAVX_NoVLX,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR128RegClassID,
        // (fsub:{ *:[v2f64] } VR128:{ *:[v2f64] }:$src1, VR128:{ *:[v2f64] }:$src2)  =>  (VSUBPDrr:{ *:[v2f64] } VR128:{ *:[v2f64] }:$src1, VR128:{ *:[v2f64] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VSUBPDrr,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 1607,
        GIR_Done,
      // Label 919: @24394
      GIM_Try, /*On fail goto*//*Label 920*/ 24417, // Rule ID 1615 //
        GIM_CheckFeatures, GIFBS_UseSSE2,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR128RegClassID,
        // (fsub:{ *:[v2f64] } VR128:{ *:[v2f64] }:$src1, VR128:{ *:[v2f64] }:$src2)  =>  (SUBPDrr:{ *:[v2f64] } VR128:{ *:[v2f64] }:$src1, VR128:{ *:[v2f64] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::SUBPDrr,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 1615,
        GIR_Done,
      // Label 920: @24417
      GIM_Try, /*On fail goto*//*Label 921*/ 24440, // Rule ID 5356 //
        GIM_CheckFeatures, GIFBS_HasAVX512_HasVLX,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128XRegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR128XRegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR128XRegClassID,
        // (fsub:{ *:[v2f64] } VR128X:{ *:[v2f64] }:$src1, VR128X:{ *:[v2f64] }:$src2)  =>  (VSUBPDZ128rr:{ *:[v2f64] } VR128X:{ *:[v2f64] }:$src1, VR128X:{ *:[v2f64] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VSUBPDZ128rr,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 5356,
        GIR_Done,
      // Label 921: @24440
      GIM_Reject,
    // Label 918: @24441
    GIM_Reject,
    // Label 901: @24442
    GIM_Try, /*On fail goto*//*Label 922*/ 24522,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v4s32,
      GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v4s32,
      GIM_Try, /*On fail goto*//*Label 923*/ 24475, // Rule ID 1605 //
        GIM_CheckFeatures, GIFBS_HasAVX_NoVLX,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR128RegClassID,
        // (fsub:{ *:[v4f32] } VR128:{ *:[v4f32] }:$src1, VR128:{ *:[v4f32] }:$src2)  =>  (VSUBPSrr:{ *:[v4f32] } VR128:{ *:[v4f32] }:$src1, VR128:{ *:[v4f32] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VSUBPSrr,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 1605,
        GIR_Done,
      // Label 923: @24475
      GIM_Try, /*On fail goto*//*Label 924*/ 24498, // Rule ID 1613 //
        GIM_CheckFeatures, GIFBS_UseSSE1,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR128RegClassID,
        // (fsub:{ *:[v4f32] } VR128:{ *:[v4f32] }:$src1, VR128:{ *:[v4f32] }:$src2)  =>  (SUBPSrr:{ *:[v4f32] } VR128:{ *:[v4f32] }:$src1, VR128:{ *:[v4f32] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::SUBPSrr,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 1613,
        GIR_Done,
      // Label 924: @24498
      GIM_Try, /*On fail goto*//*Label 925*/ 24521, // Rule ID 5338 //
        GIM_CheckFeatures, GIFBS_HasAVX512_HasVLX,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128XRegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR128XRegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR128XRegClassID,
        // (fsub:{ *:[v4f32] } VR128X:{ *:[v4f32] }:$src1, VR128X:{ *:[v4f32] }:$src2)  =>  (VSUBPSZ128rr:{ *:[v4f32] } VR128X:{ *:[v4f32] }:$src1, VR128X:{ *:[v4f32] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VSUBPSZ128rr,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 5338,
        GIR_Done,
      // Label 925: @24521
      GIM_Reject,
    // Label 922: @24522
    GIM_Reject,
    // Label 902: @24523
    GIM_Try, /*On fail goto*//*Label 926*/ 24580,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v4s64,
      GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v4s64,
      GIM_Try, /*On fail goto*//*Label 927*/ 24556, // Rule ID 1611 //
        GIM_CheckFeatures, GIFBS_HasAVX_NoVLX,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR256RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR256RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR256RegClassID,
        // (fsub:{ *:[v4f64] } VR256:{ *:[v4f64] }:$src1, VR256:{ *:[v4f64] }:$src2)  =>  (VSUBPDYrr:{ *:[v4f64] } VR256:{ *:[v4f64] }:$src1, VR256:{ *:[v4f64] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VSUBPDYrr,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 1611,
        GIR_Done,
      // Label 927: @24556
      GIM_Try, /*On fail goto*//*Label 928*/ 24579, // Rule ID 5365 //
        GIM_CheckFeatures, GIFBS_HasAVX512_HasVLX,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR256XRegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR256XRegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR256XRegClassID,
        // (fsub:{ *:[v4f64] } VR256X:{ *:[v4f64] }:$src1, VR256X:{ *:[v4f64] }:$src2)  =>  (VSUBPDZ256rr:{ *:[v4f64] } VR256X:{ *:[v4f64] }:$src1, VR256X:{ *:[v4f64] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VSUBPDZ256rr,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 5365,
        GIR_Done,
      // Label 928: @24579
      GIM_Reject,
    // Label 926: @24580
    GIM_Reject,
    // Label 903: @24581
    GIM_Try, /*On fail goto*//*Label 929*/ 24638,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v8s32,
      GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v8s32,
      GIM_Try, /*On fail goto*//*Label 930*/ 24614, // Rule ID 1609 //
        GIM_CheckFeatures, GIFBS_HasAVX_NoVLX,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR256RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR256RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR256RegClassID,
        // (fsub:{ *:[v8f32] } VR256:{ *:[v8f32] }:$src1, VR256:{ *:[v8f32] }:$src2)  =>  (VSUBPSYrr:{ *:[v8f32] } VR256:{ *:[v8f32] }:$src1, VR256:{ *:[v8f32] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VSUBPSYrr,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 1609,
        GIR_Done,
      // Label 930: @24614
      GIM_Try, /*On fail goto*//*Label 931*/ 24637, // Rule ID 5347 //
        GIM_CheckFeatures, GIFBS_HasAVX512_HasVLX,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR256XRegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR256XRegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR256XRegClassID,
        // (fsub:{ *:[v8f32] } VR256X:{ *:[v8f32] }:$src1, VR256X:{ *:[v8f32] }:$src2)  =>  (VSUBPSZ256rr:{ *:[v8f32] } VR256X:{ *:[v8f32] }:$src1, VR256X:{ *:[v8f32] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VSUBPSZ256rr,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 5347,
        GIR_Done,
      // Label 931: @24637
      GIM_Reject,
    // Label 929: @24638
    GIM_Reject,
    // Label 904: @24639
    GIM_Try, /*On fail goto*//*Label 932*/ 24670, // Rule ID 5329 //
      GIM_CheckFeatures, GIFBS_HasAVX512,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v8s64,
      GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v8s64,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR512RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR512RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR512RegClassID,
      // (fsub:{ *:[v8f64] } VR512:{ *:[v8f64] }:$src1, VR512:{ *:[v8f64] }:$src2)  =>  (VSUBPDZrr:{ *:[v8f64] } VR512:{ *:[v8f64] }:$src1, VR512:{ *:[v8f64] }:$src2)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VSUBPDZrr,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 5329,
      GIR_Done,
    // Label 932: @24670
    GIM_Reject,
    // Label 905: @24671
    GIM_Try, /*On fail goto*//*Label 933*/ 24702, // Rule ID 5320 //
      GIM_CheckFeatures, GIFBS_HasAVX512,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v16s32,
      GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v16s32,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR512RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR512RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR512RegClassID,
      // (fsub:{ *:[v16f32] } VR512:{ *:[v16f32] }:$src1, VR512:{ *:[v16f32] }:$src2)  =>  (VSUBPSZrr:{ *:[v16f32] } VR512:{ *:[v16f32] }:$src1, VR512:{ *:[v16f32] }:$src2)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VSUBPSZrr,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 5320,
      GIR_Done,
    // Label 933: @24702
    GIM_Reject,
    // Label 906: @24703
    GIM_Reject,
    // Label 20: @24704
    GIM_SwitchType, /*MI*/0, /*Op*/0, /*[*/3, 20, /*)*//*default:*//*Label 943*/ 25316,
    /*GILLT_s32*//*Label 934*/ 24727,
    /*GILLT_s64*//*Label 935*/ 24834,
    /*GILLT_s80*//*Label 936*/ 24941, 0, 0,
    /*GILLT_v2s64*//*Label 937*/ 24974, 0,
    /*GILLT_v4s32*//*Label 938*/ 25055,
    /*GILLT_v4s64*//*Label 939*/ 25136, 0, 0,
    /*GILLT_v8s32*//*Label 940*/ 25194,
    /*GILLT_v8s64*//*Label 941*/ 25252, 0, 0, 0,
    /*GILLT_v16s32*//*Label 942*/ 25284,
    // Label 934: @24727
    GIM_Try, /*On fail goto*//*Label 944*/ 24833,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_s32,
      GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_s32,
      GIM_Try, /*On fail goto*//*Label 945*/ 24763, // Rule ID 746 //
        GIM_CheckFeatures, GIFBS_FPStackf32,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::RFP32RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::RFP32RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::RFP32RegClassID,
        // (fmul:{ *:[f32] } RFP32:{ *:[f32] }:$src1, RFP32:{ *:[f32] }:$src2)  =>  (MUL_Fp32:{ *:[f32] }:{ *:[i16] } RFP32:{ *:[f32] }:$src1, RFP32:{ *:[f32] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::MUL_Fp32,
        GIR_AddImplicitDef, /*InsnID*/0, X86::FPSW,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 746,
        GIR_Done,
      // Label 945: @24763
      GIM_Try, /*On fail goto*//*Label 946*/ 24786, // Rule ID 1597 //
        GIM_CheckFeatures, GIFBS_UseAVX,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::FR32RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::FR32RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::FR32RegClassID,
        // (fmul:{ *:[f32] } FR32:{ *:[f32] }:$src1, FR32:{ *:[f32] }:$src2)  =>  (VMULSSrr:{ *:[f32] } FR32:{ *:[f32] }:$src1, FR32:{ *:[f32] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VMULSSrr,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 1597,
        GIR_Done,
      // Label 946: @24786
      GIM_Try, /*On fail goto*//*Label 947*/ 24809, // Rule ID 1601 //
        GIM_CheckFeatures, GIFBS_UseSSE1,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::FR32RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::FR32RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::FR32RegClassID,
        // (fmul:{ *:[f32] } FR32:{ *:[f32] }:$src1, FR32:{ *:[f32] }:$src2)  =>  (MULSSrr:{ *:[f32] } FR32:{ *:[f32] }:$src1, FR32:{ *:[f32] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::MULSSrr,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 1601,
        GIR_Done,
      // Label 947: @24809
      GIM_Try, /*On fail goto*//*Label 948*/ 24832, // Rule ID 5088 //
        GIM_CheckFeatures, GIFBS_HasAVX512,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::FR32XRegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::FR32XRegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::FR32XRegClassID,
        // (fmul:{ *:[f32] } FR32X:{ *:[f32] }:$src1, FR32X:{ *:[f32] }:$src2)  =>  (VMULSSZrr:{ *:[f32] } FR32X:{ *:[f32] }:$src1, FR32X:{ *:[f32] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VMULSSZrr,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 5088,
        GIR_Done,
      // Label 948: @24832
      GIM_Reject,
    // Label 944: @24833
    GIM_Reject,
    // Label 935: @24834
    GIM_Try, /*On fail goto*//*Label 949*/ 24940,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_s64,
      GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_s64,
      GIM_Try, /*On fail goto*//*Label 950*/ 24870, // Rule ID 747 //
        GIM_CheckFeatures, GIFBS_FPStackf64,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::RFP64RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::RFP64RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::RFP64RegClassID,
        // (fmul:{ *:[f64] } RFP64:{ *:[f64] }:$src1, RFP64:{ *:[f64] }:$src2)  =>  (MUL_Fp64:{ *:[f64] }:{ *:[i16] } RFP64:{ *:[f64] }:$src1, RFP64:{ *:[f64] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::MUL_Fp64,
        GIR_AddImplicitDef, /*InsnID*/0, X86::FPSW,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 747,
        GIR_Done,
      // Label 950: @24870
      GIM_Try, /*On fail goto*//*Label 951*/ 24893, // Rule ID 1599 //
        GIM_CheckFeatures, GIFBS_UseAVX,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::FR64RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::FR64RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::FR64RegClassID,
        // (fmul:{ *:[f64] } FR64:{ *:[f64] }:$src1, FR64:{ *:[f64] }:$src2)  =>  (VMULSDrr:{ *:[f64] } FR64:{ *:[f64] }:$src1, FR64:{ *:[f64] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VMULSDrr,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 1599,
        GIR_Done,
      // Label 951: @24893
      GIM_Try, /*On fail goto*//*Label 952*/ 24916, // Rule ID 1603 //
        GIM_CheckFeatures, GIFBS_UseSSE2,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::FR64RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::FR64RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::FR64RegClassID,
        // (fmul:{ *:[f64] } FR64:{ *:[f64] }:$src1, FR64:{ *:[f64] }:$src2)  =>  (MULSDrr:{ *:[f64] } FR64:{ *:[f64] }:$src1, FR64:{ *:[f64] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::MULSDrr,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 1603,
        GIR_Done,
      // Label 952: @24916
      GIM_Try, /*On fail goto*//*Label 953*/ 24939, // Rule ID 5099 //
        GIM_CheckFeatures, GIFBS_HasAVX512,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::FR64XRegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::FR64XRegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::FR64XRegClassID,
        // (fmul:{ *:[f64] } FR64X:{ *:[f64] }:$src1, FR64X:{ *:[f64] }:$src2)  =>  (VMULSDZrr:{ *:[f64] } FR64X:{ *:[f64] }:$src1, FR64X:{ *:[f64] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VMULSDZrr,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 5099,
        GIR_Done,
      // Label 953: @24939
      GIM_Reject,
    // Label 949: @24940
    GIM_Reject,
    // Label 936: @24941
    GIM_Try, /*On fail goto*//*Label 954*/ 24973, // Rule ID 748 //
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_s80,
      GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_s80,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::RFP80RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::RFP80RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::RFP80RegClassID,
      // (fmul:{ *:[f80] } RFP80:{ *:[f80] }:$src1, RFP80:{ *:[f80] }:$src2)  =>  (MUL_Fp80:{ *:[f80] }:{ *:[i16] } RFP80:{ *:[f80] }:$src1, RFP80:{ *:[f80] }:$src2)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::MUL_Fp80,
      GIR_AddImplicitDef, /*InsnID*/0, X86::FPSW,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 748,
      GIR_Done,
    // Label 954: @24973
    GIM_Reject,
    // Label 937: @24974
    GIM_Try, /*On fail goto*//*Label 955*/ 25054,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v2s64,
      GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v2s64,
      GIM_Try, /*On fail goto*//*Label 956*/ 25007, // Rule ID 1587 //
        GIM_CheckFeatures, GIFBS_HasAVX_NoVLX,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR128RegClassID,
        // (fmul:{ *:[v2f64] } VR128:{ *:[v2f64] }:$src1, VR128:{ *:[v2f64] }:$src2)  =>  (VMULPDrr:{ *:[v2f64] } VR128:{ *:[v2f64] }:$src1, VR128:{ *:[v2f64] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VMULPDrr,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 1587,
        GIR_Done,
      // Label 956: @25007
      GIM_Try, /*On fail goto*//*Label 957*/ 25030, // Rule ID 1595 //
        GIM_CheckFeatures, GIFBS_UseSSE2,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR128RegClassID,
        // (fmul:{ *:[v2f64] } VR128:{ *:[v2f64] }:$src1, VR128:{ *:[v2f64] }:$src2)  =>  (MULPDrr:{ *:[v2f64] } VR128:{ *:[v2f64] }:$src1, VR128:{ *:[v2f64] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::MULPDrr,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 1595,
        GIR_Done,
      // Label 957: @25030
      GIM_Try, /*On fail goto*//*Label 958*/ 25053, // Rule ID 5296 //
        GIM_CheckFeatures, GIFBS_HasAVX512_HasVLX,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128XRegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR128XRegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR128XRegClassID,
        // (fmul:{ *:[v2f64] } VR128X:{ *:[v2f64] }:$src1, VR128X:{ *:[v2f64] }:$src2)  =>  (VMULPDZ128rr:{ *:[v2f64] } VR128X:{ *:[v2f64] }:$src1, VR128X:{ *:[v2f64] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VMULPDZ128rr,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 5296,
        GIR_Done,
      // Label 958: @25053
      GIM_Reject,
    // Label 955: @25054
    GIM_Reject,
    // Label 938: @25055
    GIM_Try, /*On fail goto*//*Label 959*/ 25135,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v4s32,
      GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v4s32,
      GIM_Try, /*On fail goto*//*Label 960*/ 25088, // Rule ID 1585 //
        GIM_CheckFeatures, GIFBS_HasAVX_NoVLX,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR128RegClassID,
        // (fmul:{ *:[v4f32] } VR128:{ *:[v4f32] }:$src1, VR128:{ *:[v4f32] }:$src2)  =>  (VMULPSrr:{ *:[v4f32] } VR128:{ *:[v4f32] }:$src1, VR128:{ *:[v4f32] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VMULPSrr,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 1585,
        GIR_Done,
      // Label 960: @25088
      GIM_Try, /*On fail goto*//*Label 961*/ 25111, // Rule ID 1593 //
        GIM_CheckFeatures, GIFBS_UseSSE1,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR128RegClassID,
        // (fmul:{ *:[v4f32] } VR128:{ *:[v4f32] }:$src1, VR128:{ *:[v4f32] }:$src2)  =>  (MULPSrr:{ *:[v4f32] } VR128:{ *:[v4f32] }:$src1, VR128:{ *:[v4f32] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::MULPSrr,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 1593,
        GIR_Done,
      // Label 961: @25111
      GIM_Try, /*On fail goto*//*Label 962*/ 25134, // Rule ID 5278 //
        GIM_CheckFeatures, GIFBS_HasAVX512_HasVLX,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128XRegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR128XRegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR128XRegClassID,
        // (fmul:{ *:[v4f32] } VR128X:{ *:[v4f32] }:$src1, VR128X:{ *:[v4f32] }:$src2)  =>  (VMULPSZ128rr:{ *:[v4f32] } VR128X:{ *:[v4f32] }:$src1, VR128X:{ *:[v4f32] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VMULPSZ128rr,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 5278,
        GIR_Done,
      // Label 962: @25134
      GIM_Reject,
    // Label 959: @25135
    GIM_Reject,
    // Label 939: @25136
    GIM_Try, /*On fail goto*//*Label 963*/ 25193,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v4s64,
      GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v4s64,
      GIM_Try, /*On fail goto*//*Label 964*/ 25169, // Rule ID 1591 //
        GIM_CheckFeatures, GIFBS_HasAVX_NoVLX,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR256RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR256RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR256RegClassID,
        // (fmul:{ *:[v4f64] } VR256:{ *:[v4f64] }:$src1, VR256:{ *:[v4f64] }:$src2)  =>  (VMULPDYrr:{ *:[v4f64] } VR256:{ *:[v4f64] }:$src1, VR256:{ *:[v4f64] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VMULPDYrr,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 1591,
        GIR_Done,
      // Label 964: @25169
      GIM_Try, /*On fail goto*//*Label 965*/ 25192, // Rule ID 5305 //
        GIM_CheckFeatures, GIFBS_HasAVX512_HasVLX,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR256XRegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR256XRegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR256XRegClassID,
        // (fmul:{ *:[v4f64] } VR256X:{ *:[v4f64] }:$src1, VR256X:{ *:[v4f64] }:$src2)  =>  (VMULPDZ256rr:{ *:[v4f64] } VR256X:{ *:[v4f64] }:$src1, VR256X:{ *:[v4f64] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VMULPDZ256rr,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 5305,
        GIR_Done,
      // Label 965: @25192
      GIM_Reject,
    // Label 963: @25193
    GIM_Reject,
    // Label 940: @25194
    GIM_Try, /*On fail goto*//*Label 966*/ 25251,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v8s32,
      GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v8s32,
      GIM_Try, /*On fail goto*//*Label 967*/ 25227, // Rule ID 1589 //
        GIM_CheckFeatures, GIFBS_HasAVX_NoVLX,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR256RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR256RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR256RegClassID,
        // (fmul:{ *:[v8f32] } VR256:{ *:[v8f32] }:$src1, VR256:{ *:[v8f32] }:$src2)  =>  (VMULPSYrr:{ *:[v8f32] } VR256:{ *:[v8f32] }:$src1, VR256:{ *:[v8f32] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VMULPSYrr,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 1589,
        GIR_Done,
      // Label 967: @25227
      GIM_Try, /*On fail goto*//*Label 968*/ 25250, // Rule ID 5287 //
        GIM_CheckFeatures, GIFBS_HasAVX512_HasVLX,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR256XRegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR256XRegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR256XRegClassID,
        // (fmul:{ *:[v8f32] } VR256X:{ *:[v8f32] }:$src1, VR256X:{ *:[v8f32] }:$src2)  =>  (VMULPSZ256rr:{ *:[v8f32] } VR256X:{ *:[v8f32] }:$src1, VR256X:{ *:[v8f32] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VMULPSZ256rr,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 5287,
        GIR_Done,
      // Label 968: @25250
      GIM_Reject,
    // Label 966: @25251
    GIM_Reject,
    // Label 941: @25252
    GIM_Try, /*On fail goto*//*Label 969*/ 25283, // Rule ID 5269 //
      GIM_CheckFeatures, GIFBS_HasAVX512,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v8s64,
      GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v8s64,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR512RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR512RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR512RegClassID,
      // (fmul:{ *:[v8f64] } VR512:{ *:[v8f64] }:$src1, VR512:{ *:[v8f64] }:$src2)  =>  (VMULPDZrr:{ *:[v8f64] } VR512:{ *:[v8f64] }:$src1, VR512:{ *:[v8f64] }:$src2)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VMULPDZrr,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 5269,
      GIR_Done,
    // Label 969: @25283
    GIM_Reject,
    // Label 942: @25284
    GIM_Try, /*On fail goto*//*Label 970*/ 25315, // Rule ID 5260 //
      GIM_CheckFeatures, GIFBS_HasAVX512,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v16s32,
      GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v16s32,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR512RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR512RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR512RegClassID,
      // (fmul:{ *:[v16f32] } VR512:{ *:[v16f32] }:$src1, VR512:{ *:[v16f32] }:$src2)  =>  (VMULPSZrr:{ *:[v16f32] } VR512:{ *:[v16f32] }:$src1, VR512:{ *:[v16f32] }:$src2)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VMULPSZrr,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 5260,
      GIR_Done,
    // Label 970: @25315
    GIM_Reject,
    // Label 943: @25316
    GIM_Reject,
    // Label 21: @25317
    GIM_SwitchType, /*MI*/0, /*Op*/0, /*[*/3, 20, /*)*//*default:*//*Label 980*/ 25929,
    /*GILLT_s32*//*Label 971*/ 25340,
    /*GILLT_s64*//*Label 972*/ 25447,
    /*GILLT_s80*//*Label 973*/ 25554, 0, 0,
    /*GILLT_v2s64*//*Label 974*/ 25587, 0,
    /*GILLT_v4s32*//*Label 975*/ 25668,
    /*GILLT_v4s64*//*Label 976*/ 25749, 0, 0,
    /*GILLT_v8s32*//*Label 977*/ 25807,
    /*GILLT_v8s64*//*Label 978*/ 25865, 0, 0, 0,
    /*GILLT_v16s32*//*Label 979*/ 25897,
    // Label 971: @25340
    GIM_Try, /*On fail goto*//*Label 981*/ 25446,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_s32,
      GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_s32,
      GIM_Try, /*On fail goto*//*Label 982*/ 25376, // Rule ID 749 //
        GIM_CheckFeatures, GIFBS_FPStackf32,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::RFP32RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::RFP32RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::RFP32RegClassID,
        // (fdiv:{ *:[f32] } RFP32:{ *:[f32] }:$src1, RFP32:{ *:[f32] }:$src2)  =>  (DIV_Fp32:{ *:[f32] }:{ *:[i16] } RFP32:{ *:[f32] }:$src1, RFP32:{ *:[f32] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::DIV_Fp32,
        GIR_AddImplicitDef, /*InsnID*/0, X86::FPSW,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 749,
        GIR_Done,
      // Label 982: @25376
      GIM_Try, /*On fail goto*//*Label 983*/ 25399, // Rule ID 1637 //
        GIM_CheckFeatures, GIFBS_UseAVX,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::FR32RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::FR32RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::FR32RegClassID,
        // (fdiv:{ *:[f32] } FR32:{ *:[f32] }:$src1, FR32:{ *:[f32] }:$src2)  =>  (VDIVSSrr:{ *:[f32] } FR32:{ *:[f32] }:$src1, FR32:{ *:[f32] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VDIVSSrr,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 1637,
        GIR_Done,
      // Label 983: @25399
      GIM_Try, /*On fail goto*//*Label 984*/ 25422, // Rule ID 1641 //
        GIM_CheckFeatures, GIFBS_UseSSE1,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::FR32RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::FR32RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::FR32RegClassID,
        // (fdiv:{ *:[f32] } FR32:{ *:[f32] }:$src1, FR32:{ *:[f32] }:$src2)  =>  (DIVSSrr:{ *:[f32] } FR32:{ *:[f32] }:$src1, FR32:{ *:[f32] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::DIVSSrr,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 1641,
        GIR_Done,
      // Label 984: @25422
      GIM_Try, /*On fail goto*//*Label 985*/ 25445, // Rule ID 5132 //
        GIM_CheckFeatures, GIFBS_HasAVX512,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::FR32XRegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::FR32XRegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::FR32XRegClassID,
        // (fdiv:{ *:[f32] } FR32X:{ *:[f32] }:$src1, FR32X:{ *:[f32] }:$src2)  =>  (VDIVSSZrr:{ *:[f32] } FR32X:{ *:[f32] }:$src1, FR32X:{ *:[f32] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VDIVSSZrr,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 5132,
        GIR_Done,
      // Label 985: @25445
      GIM_Reject,
    // Label 981: @25446
    GIM_Reject,
    // Label 972: @25447
    GIM_Try, /*On fail goto*//*Label 986*/ 25553,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_s64,
      GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_s64,
      GIM_Try, /*On fail goto*//*Label 987*/ 25483, // Rule ID 750 //
        GIM_CheckFeatures, GIFBS_FPStackf64,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::RFP64RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::RFP64RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::RFP64RegClassID,
        // (fdiv:{ *:[f64] } RFP64:{ *:[f64] }:$src1, RFP64:{ *:[f64] }:$src2)  =>  (DIV_Fp64:{ *:[f64] }:{ *:[i16] } RFP64:{ *:[f64] }:$src1, RFP64:{ *:[f64] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::DIV_Fp64,
        GIR_AddImplicitDef, /*InsnID*/0, X86::FPSW,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 750,
        GIR_Done,
      // Label 987: @25483
      GIM_Try, /*On fail goto*//*Label 988*/ 25506, // Rule ID 1639 //
        GIM_CheckFeatures, GIFBS_UseAVX,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::FR64RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::FR64RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::FR64RegClassID,
        // (fdiv:{ *:[f64] } FR64:{ *:[f64] }:$src1, FR64:{ *:[f64] }:$src2)  =>  (VDIVSDrr:{ *:[f64] } FR64:{ *:[f64] }:$src1, FR64:{ *:[f64] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VDIVSDrr,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 1639,
        GIR_Done,
      // Label 988: @25506
      GIM_Try, /*On fail goto*//*Label 989*/ 25529, // Rule ID 1643 //
        GIM_CheckFeatures, GIFBS_UseSSE2,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::FR64RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::FR64RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::FR64RegClassID,
        // (fdiv:{ *:[f64] } FR64:{ *:[f64] }:$src1, FR64:{ *:[f64] }:$src2)  =>  (DIVSDrr:{ *:[f64] } FR64:{ *:[f64] }:$src1, FR64:{ *:[f64] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::DIVSDrr,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 1643,
        GIR_Done,
      // Label 989: @25529
      GIM_Try, /*On fail goto*//*Label 990*/ 25552, // Rule ID 5143 //
        GIM_CheckFeatures, GIFBS_HasAVX512,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::FR64XRegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::FR64XRegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::FR64XRegClassID,
        // (fdiv:{ *:[f64] } FR64X:{ *:[f64] }:$src1, FR64X:{ *:[f64] }:$src2)  =>  (VDIVSDZrr:{ *:[f64] } FR64X:{ *:[f64] }:$src1, FR64X:{ *:[f64] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VDIVSDZrr,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 5143,
        GIR_Done,
      // Label 990: @25552
      GIM_Reject,
    // Label 986: @25553
    GIM_Reject,
    // Label 973: @25554
    GIM_Try, /*On fail goto*//*Label 991*/ 25586, // Rule ID 751 //
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_s80,
      GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_s80,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::RFP80RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::RFP80RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::RFP80RegClassID,
      // (fdiv:{ *:[f80] } RFP80:{ *:[f80] }:$src1, RFP80:{ *:[f80] }:$src2)  =>  (DIV_Fp80:{ *:[f80] }:{ *:[i16] } RFP80:{ *:[f80] }:$src1, RFP80:{ *:[f80] }:$src2)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::DIV_Fp80,
      GIR_AddImplicitDef, /*InsnID*/0, X86::FPSW,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 751,
      GIR_Done,
    // Label 991: @25586
    GIM_Reject,
    // Label 974: @25587
    GIM_Try, /*On fail goto*//*Label 992*/ 25667,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v2s64,
      GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v2s64,
      GIM_Try, /*On fail goto*//*Label 993*/ 25620, // Rule ID 1627 //
        GIM_CheckFeatures, GIFBS_HasAVX_NoVLX,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR128RegClassID,
        // (fdiv:{ *:[v2f64] } VR128:{ *:[v2f64] }:$src1, VR128:{ *:[v2f64] }:$src2)  =>  (VDIVPDrr:{ *:[v2f64] } VR128:{ *:[v2f64] }:$src1, VR128:{ *:[v2f64] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VDIVPDrr,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 1627,
        GIR_Done,
      // Label 993: @25620
      GIM_Try, /*On fail goto*//*Label 994*/ 25643, // Rule ID 1635 //
        GIM_CheckFeatures, GIFBS_UseSSE2,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR128RegClassID,
        // (fdiv:{ *:[v2f64] } VR128:{ *:[v2f64] }:$src1, VR128:{ *:[v2f64] }:$src2)  =>  (DIVPDrr:{ *:[v2f64] } VR128:{ *:[v2f64] }:$src1, VR128:{ *:[v2f64] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::DIVPDrr,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 1635,
        GIR_Done,
      // Label 994: @25643
      GIM_Try, /*On fail goto*//*Label 995*/ 25666, // Rule ID 5416 //
        GIM_CheckFeatures, GIFBS_HasAVX512_HasVLX,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128XRegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR128XRegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR128XRegClassID,
        // (fdiv:{ *:[v2f64] } VR128X:{ *:[v2f64] }:$src1, VR128X:{ *:[v2f64] }:$src2)  =>  (VDIVPDZ128rr:{ *:[v2f64] } VR128X:{ *:[v2f64] }:$src1, VR128X:{ *:[v2f64] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VDIVPDZ128rr,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 5416,
        GIR_Done,
      // Label 995: @25666
      GIM_Reject,
    // Label 992: @25667
    GIM_Reject,
    // Label 975: @25668
    GIM_Try, /*On fail goto*//*Label 996*/ 25748,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v4s32,
      GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v4s32,
      GIM_Try, /*On fail goto*//*Label 997*/ 25701, // Rule ID 1625 //
        GIM_CheckFeatures, GIFBS_HasAVX_NoVLX,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR128RegClassID,
        // (fdiv:{ *:[v4f32] } VR128:{ *:[v4f32] }:$src1, VR128:{ *:[v4f32] }:$src2)  =>  (VDIVPSrr:{ *:[v4f32] } VR128:{ *:[v4f32] }:$src1, VR128:{ *:[v4f32] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VDIVPSrr,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 1625,
        GIR_Done,
      // Label 997: @25701
      GIM_Try, /*On fail goto*//*Label 998*/ 25724, // Rule ID 1633 //
        GIM_CheckFeatures, GIFBS_UseSSE1,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR128RegClassID,
        // (fdiv:{ *:[v4f32] } VR128:{ *:[v4f32] }:$src1, VR128:{ *:[v4f32] }:$src2)  =>  (DIVPSrr:{ *:[v4f32] } VR128:{ *:[v4f32] }:$src1, VR128:{ *:[v4f32] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::DIVPSrr,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 1633,
        GIR_Done,
      // Label 998: @25724
      GIM_Try, /*On fail goto*//*Label 999*/ 25747, // Rule ID 5398 //
        GIM_CheckFeatures, GIFBS_HasAVX512_HasVLX,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128XRegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR128XRegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR128XRegClassID,
        // (fdiv:{ *:[v4f32] } VR128X:{ *:[v4f32] }:$src1, VR128X:{ *:[v4f32] }:$src2)  =>  (VDIVPSZ128rr:{ *:[v4f32] } VR128X:{ *:[v4f32] }:$src1, VR128X:{ *:[v4f32] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VDIVPSZ128rr,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 5398,
        GIR_Done,
      // Label 999: @25747
      GIM_Reject,
    // Label 996: @25748
    GIM_Reject,
    // Label 976: @25749
    GIM_Try, /*On fail goto*//*Label 1000*/ 25806,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v4s64,
      GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v4s64,
      GIM_Try, /*On fail goto*//*Label 1001*/ 25782, // Rule ID 1631 //
        GIM_CheckFeatures, GIFBS_HasAVX_NoVLX,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR256RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR256RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR256RegClassID,
        // (fdiv:{ *:[v4f64] } VR256:{ *:[v4f64] }:$src1, VR256:{ *:[v4f64] }:$src2)  =>  (VDIVPDYrr:{ *:[v4f64] } VR256:{ *:[v4f64] }:$src1, VR256:{ *:[v4f64] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VDIVPDYrr,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 1631,
        GIR_Done,
      // Label 1001: @25782
      GIM_Try, /*On fail goto*//*Label 1002*/ 25805, // Rule ID 5425 //
        GIM_CheckFeatures, GIFBS_HasAVX512_HasVLX,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR256XRegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR256XRegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR256XRegClassID,
        // (fdiv:{ *:[v4f64] } VR256X:{ *:[v4f64] }:$src1, VR256X:{ *:[v4f64] }:$src2)  =>  (VDIVPDZ256rr:{ *:[v4f64] } VR256X:{ *:[v4f64] }:$src1, VR256X:{ *:[v4f64] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VDIVPDZ256rr,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 5425,
        GIR_Done,
      // Label 1002: @25805
      GIM_Reject,
    // Label 1000: @25806
    GIM_Reject,
    // Label 977: @25807
    GIM_Try, /*On fail goto*//*Label 1003*/ 25864,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v8s32,
      GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v8s32,
      GIM_Try, /*On fail goto*//*Label 1004*/ 25840, // Rule ID 1629 //
        GIM_CheckFeatures, GIFBS_HasAVX_NoVLX,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR256RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR256RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR256RegClassID,
        // (fdiv:{ *:[v8f32] } VR256:{ *:[v8f32] }:$src1, VR256:{ *:[v8f32] }:$src2)  =>  (VDIVPSYrr:{ *:[v8f32] } VR256:{ *:[v8f32] }:$src1, VR256:{ *:[v8f32] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VDIVPSYrr,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 1629,
        GIR_Done,
      // Label 1004: @25840
      GIM_Try, /*On fail goto*//*Label 1005*/ 25863, // Rule ID 5407 //
        GIM_CheckFeatures, GIFBS_HasAVX512_HasVLX,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR256XRegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR256XRegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR256XRegClassID,
        // (fdiv:{ *:[v8f32] } VR256X:{ *:[v8f32] }:$src1, VR256X:{ *:[v8f32] }:$src2)  =>  (VDIVPSZ256rr:{ *:[v8f32] } VR256X:{ *:[v8f32] }:$src1, VR256X:{ *:[v8f32] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VDIVPSZ256rr,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 5407,
        GIR_Done,
      // Label 1005: @25863
      GIM_Reject,
    // Label 1003: @25864
    GIM_Reject,
    // Label 978: @25865
    GIM_Try, /*On fail goto*//*Label 1006*/ 25896, // Rule ID 5389 //
      GIM_CheckFeatures, GIFBS_HasAVX512,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v8s64,
      GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v8s64,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR512RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR512RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR512RegClassID,
      // (fdiv:{ *:[v8f64] } VR512:{ *:[v8f64] }:$src1, VR512:{ *:[v8f64] }:$src2)  =>  (VDIVPDZrr:{ *:[v8f64] } VR512:{ *:[v8f64] }:$src1, VR512:{ *:[v8f64] }:$src2)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VDIVPDZrr,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 5389,
      GIR_Done,
    // Label 1006: @25896
    GIM_Reject,
    // Label 979: @25897
    GIM_Try, /*On fail goto*//*Label 1007*/ 25928, // Rule ID 5380 //
      GIM_CheckFeatures, GIFBS_HasAVX512,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v16s32,
      GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v16s32,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR512RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR512RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR512RegClassID,
      // (fdiv:{ *:[v16f32] } VR512:{ *:[v16f32] }:$src1, VR512:{ *:[v16f32] }:$src2)  =>  (VDIVPSZrr:{ *:[v16f32] } VR512:{ *:[v16f32] }:$src1, VR512:{ *:[v16f32] }:$src2)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VDIVPSZrr,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 5380,
      GIR_Done,
    // Label 1007: @25928
    GIM_Reject,
    // Label 980: @25929
    GIM_Reject,
    // Label 22: @25930
    GIM_SwitchType, /*MI*/0, /*Op*/0, /*[*/3, 6, /*)*//*default:*//*Label 1011*/ 26018,
    /*GILLT_s32*//*Label 1008*/ 25939,
    /*GILLT_s64*//*Label 1009*/ 25966,
    /*GILLT_s80*//*Label 1010*/ 25993,
    // Label 1008: @25939
    GIM_Try, /*On fail goto*//*Label 1012*/ 25965, // Rule ID 818 //
      GIM_CheckFeatures, GIFBS_FPStackf32,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_s32,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::RFP32RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::RFP32RegClassID,
      // (fneg:{ *:[f32] } RFP32:{ *:[f32] }:$src)  =>  (CHS_Fp32:{ *:[f32] }:{ *:[i16] } RFP32:{ *:[f32] }:$src)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::CHS_Fp32,
      GIR_AddImplicitDef, /*InsnID*/0, X86::FPSW,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 818,
      GIR_Done,
    // Label 1012: @25965
    GIM_Reject,
    // Label 1009: @25966
    GIM_Try, /*On fail goto*//*Label 1013*/ 25992, // Rule ID 819 //
      GIM_CheckFeatures, GIFBS_FPStackf64,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_s64,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::RFP64RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::RFP64RegClassID,
      // (fneg:{ *:[f64] } RFP64:{ *:[f64] }:$src)  =>  (CHS_Fp64:{ *:[f64] }:{ *:[i16] } RFP64:{ *:[f64] }:$src)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::CHS_Fp64,
      GIR_AddImplicitDef, /*InsnID*/0, X86::FPSW,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 819,
      GIR_Done,
    // Label 1013: @25992
    GIM_Reject,
    // Label 1010: @25993
    GIM_Try, /*On fail goto*//*Label 1014*/ 26017, // Rule ID 820 //
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_s80,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::RFP80RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::RFP80RegClassID,
      // (fneg:{ *:[f80] } RFP80:{ *:[f80] }:$src)  =>  (CHS_Fp80:{ *:[f80] }:{ *:[i16] } RFP80:{ *:[f80] }:$src)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::CHS_Fp80,
      GIR_AddImplicitDef, /*InsnID*/0, X86::FPSW,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 820,
      GIR_Done,
    // Label 1014: @26017
    GIM_Reject,
    // Label 1011: @26018
    GIM_Reject,
    // Label 23: @26019
    GIM_SwitchType, /*MI*/0, /*Op*/0, /*[*/4, 16, /*)*//*default:*//*Label 1019*/ 26294,
    /*GILLT_s64*//*Label 1015*/ 26037,
    /*GILLT_s80*//*Label 1016*/ 26173, 0, 0, 0, 0, 0,
    /*GILLT_v4s64*//*Label 1017*/ 26224, 0, 0, 0,
    /*GILLT_v8s64*//*Label 1018*/ 26270,
    // Label 1015: @26037
    GIM_Try, /*On fail goto*//*Label 1020*/ 26172,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_s32,
      GIM_Try, /*On fail goto*//*Label 1021*/ 26062, // Rule ID 1403 //
        GIM_CheckFeatures, GIFBS_UseSSE2,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::FR64RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::FR32RegClassID,
        // (fpextend:{ *:[f64] } FR32:{ *:[f32] }:$src)  =>  (CVTSS2SDrr:{ *:[f64] } FR32:{ *:[f32] }:$src)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::CVTSS2SDrr,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 1403,
        GIR_Done,
      // Label 1021: @26062
      GIM_Try, /*On fail goto*//*Label 1022*/ 26083, // Rule ID 12293 //
        GIM_CheckFeatures, GIFBS_FPStackf32,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::RFP64RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::RFP32RegClassID,
        // (fpextend:{ *:[f64] } RFP32:{ *:[f32] }:$src)  =>  (COPY_TO_REGCLASS:{ *:[f64] } RFP32:{ *:[f32] }:$src, RFP64:{ *:[i32] })
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/TargetOpcode::COPY,
        GIR_ConstrainOperandRC, /*InsnID*/0, /*Op*/0, /*RC RFP64*/55,
        // GIR_Coverage, 12293,
        GIR_Done,
      // Label 1022: @26083
      GIM_Try, /*On fail goto*//*Label 1023*/ 26127, // Rule ID 12441 //
        GIM_CheckFeatures, GIFBS_UseAVX,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::FR64RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::FR32RegClassID,
        // (fpextend:{ *:[f64] } FR32:{ *:[f32] }:$src)  =>  (VCVTSS2SDrr:{ *:[f64] } (IMPLICIT_DEF:{ *:[f64] }), FR32:{ *:[f32] }:$src)
        GIR_MakeTempReg, /*TempRegID*/0, /*TypeID*/GILLT_s64,
        GIR_BuildMI, /*InsnID*/1, /*Opcode*/TargetOpcode::IMPLICIT_DEF,
        GIR_AddTempRegister, /*InsnID*/1, /*TempRegID*/0, /*TempRegFlags*/RegState::Define,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/1,
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::VCVTSS2SDrr,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_AddTempRegister, /*InsnID*/0, /*TempRegID*/0, /*TempRegFlags*/0,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // src
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 12441,
        GIR_Done,
      // Label 1023: @26127
      GIM_Try, /*On fail goto*//*Label 1024*/ 26171, // Rule ID 14812 //
        GIM_CheckFeatures, GIFBS_HasAVX512,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::FR64XRegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::FR32XRegClassID,
        // (fpextend:{ *:[f64] } FR32X:{ *:[f32] }:$src)  =>  (VCVTSS2SDZrr:{ *:[f64] } (IMPLICIT_DEF:{ *:[f64] }), FR32X:{ *:[f32] }:$src)
        GIR_MakeTempReg, /*TempRegID*/0, /*TypeID*/GILLT_s64,
        GIR_BuildMI, /*InsnID*/1, /*Opcode*/TargetOpcode::IMPLICIT_DEF,
        GIR_AddTempRegister, /*InsnID*/1, /*TempRegID*/0, /*TempRegFlags*/RegState::Define,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/1,
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::VCVTSS2SDZrr,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_AddTempRegister, /*InsnID*/0, /*TempRegID*/0, /*TempRegFlags*/0,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // src
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 14812,
        GIR_Done,
      // Label 1024: @26171
      GIM_Reject,
    // Label 1020: @26172
    GIM_Reject,
    // Label 1016: @26173
    GIM_Try, /*On fail goto*//*Label 1025*/ 26198, // Rule ID 12294 //
      GIM_CheckFeatures, GIFBS_FPStackf32,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_s32,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::RFP80RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::RFP32RegClassID,
      // (fpextend:{ *:[f80] } RFP32:{ *:[f32] }:$src)  =>  (COPY_TO_REGCLASS:{ *:[f80] } RFP32:{ *:[f32] }:$src, RFP80:{ *:[i32] })
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/TargetOpcode::COPY,
      GIR_ConstrainOperandRC, /*InsnID*/0, /*Op*/0, /*RC RFP80*/95,
      // GIR_Coverage, 12294,
      GIR_Done,
    // Label 1025: @26198
    GIM_Try, /*On fail goto*//*Label 1026*/ 26223, // Rule ID 12295 //
      GIM_CheckFeatures, GIFBS_FPStackf64,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_s64,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::RFP80RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::RFP64RegClassID,
      // (fpextend:{ *:[f80] } RFP64:{ *:[f64] }:$src)  =>  (COPY_TO_REGCLASS:{ *:[f80] } RFP64:{ *:[f64] }:$src, RFP80:{ *:[i32] })
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/TargetOpcode::COPY,
      GIR_ConstrainOperandRC, /*InsnID*/0, /*Op*/0, /*RC RFP80*/95,
      // GIR_Coverage, 12295,
      GIR_Done,
    // Label 1026: @26223
    GIM_Reject,
    // Label 1017: @26224
    GIM_Try, /*On fail goto*//*Label 1027*/ 26269,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v4s32,
      GIM_Try, /*On fail goto*//*Label 1028*/ 26249, // Rule ID 1431 //
        GIM_CheckFeatures, GIFBS_HasAVX_NoVLX,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR256RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR128RegClassID,
        // (fpextend:{ *:[v4f64] } VR128:{ *:[v4f32] }:$src)  =>  (VCVTPS2PDYrr:{ *:[v4f64] } VR128:{ *:[v4f32] }:$src)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VCVTPS2PDYrr,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 1431,
        GIR_Done,
      // Label 1028: @26249
      GIM_Try, /*On fail goto*//*Label 1029*/ 26268, // Rule ID 8155 //
        GIM_CheckFeatures, GIFBS_HasVLX,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR256XRegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR128XRegClassID,
        // (fpextend:{ *:[v4f64] } VR128X:{ *:[v4f32] }:$src)  =>  (VCVTPS2PDZ256rr:{ *:[v4f64] } VR128X:{ *:[v4f32] }:$src)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VCVTPS2PDZ256rr,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 8155,
        GIR_Done,
      // Label 1029: @26268
      GIM_Reject,
    // Label 1027: @26269
    GIM_Reject,
    // Label 1018: @26270
    GIM_Try, /*On fail goto*//*Label 1030*/ 26293, // Rule ID 8134 //
      GIM_CheckFeatures, GIFBS_HasAVX512,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v8s32,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR512RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR256XRegClassID,
      // (fpextend:{ *:[v8f64] } VR256X:{ *:[v8f32] }:$src)  =>  (VCVTPS2PDZrr:{ *:[v8f64] } VR256X:{ *:[v8f32] }:$src)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VCVTPS2PDZrr,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 8134,
      GIR_Done,
    // Label 1030: @26293
    GIM_Reject,
    // Label 1019: @26294
    GIM_Reject,
    // Label 24: @26295
    GIM_SwitchType, /*MI*/0, /*Op*/0, /*[*/3, 15, /*)*//*default:*//*Label 1035*/ 26579,
    /*GILLT_s32*//*Label 1031*/ 26313,
    /*GILLT_s64*//*Label 1032*/ 26483, 0, 0, 0, 0, 0,
    /*GILLT_v4s32*//*Label 1033*/ 26509, 0, 0, 0,
    /*GILLT_v8s32*//*Label 1034*/ 26555,
    // Label 1031: @26313
    GIM_Try, /*On fail goto*//*Label 1036*/ 26336, // Rule ID 1397 //
      GIM_CheckFeatures, GIFBS_UseSSE2,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_s64,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::FR32RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::FR64RegClassID,
      // (fpround:{ *:[f32] } FR64:{ *:[f64] }:$src)  =>  (CVTSD2SSrr:{ *:[f32] } FR64:{ *:[f64] }:$src)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::CVTSD2SSrr,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 1397,
      GIR_Done,
    // Label 1036: @26336
    GIM_Try, /*On fail goto*//*Label 1037*/ 26361, // Rule ID 12296 //
      GIM_CheckFeatures, GIFBS_FPStackf32,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_s64,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::RFP32RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::RFP64RegClassID,
      // (fpround:{ *:[f32] } RFP64:{ *:[f64] }:$src)  =>  (COPY_TO_REGCLASS:{ *:[f32] } RFP64:{ *:[f64] }:$src, RFP32:{ *:[i32] })
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/TargetOpcode::COPY,
      GIR_ConstrainOperandRC, /*InsnID*/0, /*Op*/0, /*RC RFP32*/33,
      // GIR_Coverage, 12296,
      GIR_Done,
    // Label 1037: @26361
    GIM_Try, /*On fail goto*//*Label 1038*/ 26386, // Rule ID 12297 //
      GIM_CheckFeatures, GIFBS_FPStackf32,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_s80,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::RFP32RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::RFP80RegClassID,
      // (fpround:{ *:[f32] } RFP80:{ *:[f80] }:$src)  =>  (COPY_TO_REGCLASS:{ *:[f32] } RFP80:{ *:[f80] }:$src, RFP32:{ *:[i32] })
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/TargetOpcode::COPY,
      GIR_ConstrainOperandRC, /*InsnID*/0, /*Op*/0, /*RC RFP32*/33,
      // GIR_Coverage, 12297,
      GIR_Done,
    // Label 1038: @26386
    GIM_Try, /*On fail goto*//*Label 1039*/ 26434, // Rule ID 12440 //
      GIM_CheckFeatures, GIFBS_UseAVX,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_s64,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::FR32RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::FR64RegClassID,
      // (fpround:{ *:[f32] } FR64:{ *:[f64] }:$src)  =>  (VCVTSD2SSrr:{ *:[f32] } (IMPLICIT_DEF:{ *:[f32] }), FR64:{ *:[f64] }:$src)
      GIR_MakeTempReg, /*TempRegID*/0, /*TypeID*/GILLT_s32,
      GIR_BuildMI, /*InsnID*/1, /*Opcode*/TargetOpcode::IMPLICIT_DEF,
      GIR_AddTempRegister, /*InsnID*/1, /*TempRegID*/0, /*TempRegFlags*/RegState::Define,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/1,
      GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::VCVTSD2SSrr,
      GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
      GIR_AddTempRegister, /*InsnID*/0, /*TempRegID*/0, /*TempRegFlags*/0,
      GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // src
      GIR_EraseFromParent, /*InsnID*/0,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 12440,
      GIR_Done,
    // Label 1039: @26434
    GIM_Try, /*On fail goto*//*Label 1040*/ 26482, // Rule ID 14816 //
      GIM_CheckFeatures, GIFBS_HasAVX512,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_s64,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::FR32XRegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::FR64XRegClassID,
      // (fpround:{ *:[f32] } FR64X:{ *:[f64] }:$src)  =>  (VCVTSD2SSZrr:{ *:[f32] } (IMPLICIT_DEF:{ *:[f32] }), FR64X:{ *:[f64] }:$src)
      GIR_MakeTempReg, /*TempRegID*/0, /*TypeID*/GILLT_s32,
      GIR_BuildMI, /*InsnID*/1, /*Opcode*/TargetOpcode::IMPLICIT_DEF,
      GIR_AddTempRegister, /*InsnID*/1, /*TempRegID*/0, /*TempRegFlags*/RegState::Define,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/1,
      GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::VCVTSD2SSZrr,
      GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
      GIR_AddTempRegister, /*InsnID*/0, /*TempRegID*/0, /*TempRegFlags*/0,
      GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // src
      GIR_EraseFromParent, /*InsnID*/0,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 14816,
      GIR_Done,
    // Label 1040: @26482
    GIM_Reject,
    // Label 1032: @26483
    GIM_Try, /*On fail goto*//*Label 1041*/ 26508, // Rule ID 12298 //
      GIM_CheckFeatures, GIFBS_FPStackf64,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_s80,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::RFP64RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::RFP80RegClassID,
      // (fpround:{ *:[f64] } RFP80:{ *:[f80] }:$src)  =>  (COPY_TO_REGCLASS:{ *:[f64] } RFP80:{ *:[f80] }:$src, RFP64:{ *:[i32] })
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/TargetOpcode::COPY,
      GIR_ConstrainOperandRC, /*InsnID*/0, /*Op*/0, /*RC RFP64*/55,
      // GIR_Coverage, 12298,
      GIR_Done,
    // Label 1041: @26508
    GIM_Reject,
    // Label 1033: @26509
    GIM_Try, /*On fail goto*//*Label 1042*/ 26554,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v4s64,
      GIM_Try, /*On fail goto*//*Label 1043*/ 26534, // Rule ID 1443 //
        GIM_CheckFeatures, GIFBS_HasAVX_NoVLX,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR256RegClassID,
        // (fpround:{ *:[v4f32] } VR256:{ *:[v4f64] }:$src)  =>  (VCVTPD2PSYrr:{ *:[v4f32] } VR256:{ *:[v4f64] }:$src)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VCVTPD2PSYrr,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 1443,
        GIR_Done,
      // Label 1043: @26534
      GIM_Try, /*On fail goto*//*Label 1044*/ 26553, // Rule ID 8125 //
        GIM_CheckFeatures, GIFBS_HasVLX,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128XRegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR256XRegClassID,
        // (fpround:{ *:[v4f32] } VR256X:{ *:[v4f64] }:$src)  =>  (VCVTPD2PSZ256rr:{ *:[v4f32] } VR256X:{ *:[v4f64] }:$src)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VCVTPD2PSZ256rr,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 8125,
        GIR_Done,
      // Label 1044: @26553
      GIM_Reject,
    // Label 1042: @26554
    GIM_Reject,
    // Label 1034: @26555
    GIM_Try, /*On fail goto*//*Label 1045*/ 26578, // Rule ID 8104 //
      GIM_CheckFeatures, GIFBS_HasAVX512,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v8s64,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR256XRegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR512RegClassID,
      // (fpround:{ *:[v8f32] } VR512:{ *:[v8f64] }:$src)  =>  (VCVTPD2PSZrr:{ *:[v8f32] } VR512:{ *:[v8f64] }:$src)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VCVTPD2PSZrr,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 8104,
      GIR_Done,
    // Label 1045: @26578
    GIM_Reject,
    // Label 1035: @26579
    GIM_Reject,
    // Label 25: @26580
    GIM_SwitchType, /*MI*/0, /*Op*/0, /*[*/3, 20, /*)*//*default:*//*Label 1054*/ 27209,
    /*GILLT_s32*//*Label 1046*/ 26603,
    /*GILLT_s64*//*Label 1047*/ 26742, 0, 0, 0,
    /*GILLT_v2s64*//*Label 1048*/ 26881, 0,
    /*GILLT_v4s32*//*Label 1049*/ 26905,
    /*GILLT_v4s64*//*Label 1050*/ 27021, 0, 0,
    /*GILLT_v8s32*//*Label 1051*/ 27068,
    /*GILLT_v8s64*//*Label 1052*/ 27138, 0, 0, 0,
    /*GILLT_v16s32*//*Label 1053*/ 27185,
    // Label 1046: @26603
    GIM_Try, /*On fail goto*//*Label 1055*/ 26626, // Rule ID 1335 //
      GIM_CheckFeatures, GIFBS_UseAVX,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_s32,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::GR32RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::FR32RegClassID,
      // (fp_to_sint:{ *:[i32] } FR32:{ *:[f32] }:$src)  =>  (VCVTTSS2SIrr:{ *:[i32] } FR32:{ *:[f32] }:$src)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VCVTTSS2SIrr,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 1335,
      GIR_Done,
    // Label 1055: @26626
    GIM_Try, /*On fail goto*//*Label 1056*/ 26649, // Rule ID 1339 //
      GIM_CheckFeatures, GIFBS_UseAVX,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_s64,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::GR32RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::FR64RegClassID,
      // (fp_to_sint:{ *:[i32] } FR64:{ *:[f64] }:$src)  =>  (VCVTTSD2SIrr:{ *:[i32] } FR64:{ *:[f64] }:$src)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VCVTTSD2SIrr,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 1339,
      GIR_Done,
    // Label 1056: @26649
    GIM_Try, /*On fail goto*//*Label 1057*/ 26672, // Rule ID 1343 //
      GIM_CheckFeatures, GIFBS_UseSSE1,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_s32,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::GR32RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::FR32RegClassID,
      // (fp_to_sint:{ *:[i32] } FR32:{ *:[f32] }:$src)  =>  (CVTTSS2SIrr:{ *:[i32] } FR32:{ *:[f32] }:$src)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::CVTTSS2SIrr,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 1343,
      GIR_Done,
    // Label 1057: @26672
    GIM_Try, /*On fail goto*//*Label 1058*/ 26695, // Rule ID 1347 //
      GIM_CheckFeatures, GIFBS_UseSSE2,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_s64,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::GR32RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::FR64RegClassID,
      // (fp_to_sint:{ *:[i32] } FR64:{ *:[f64] }:$src)  =>  (CVTTSD2SIrr:{ *:[i32] } FR64:{ *:[f64] }:$src)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::CVTTSD2SIrr,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 1347,
      GIR_Done,
    // Label 1058: @26695
    GIM_Try, /*On fail goto*//*Label 1059*/ 26718, // Rule ID 8046 //
      GIM_CheckFeatures, GIFBS_HasAVX512,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_s32,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::GR32RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::FR32XRegClassID,
      // (fp_to_sint:{ *:[i32] } FR32X:{ *:[f32] }:$src)  =>  (VCVTTSS2SIZrr:{ *:[i32] } FR32X:{ *:[f32] }:$src)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VCVTTSS2SIZrr,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 8046,
      GIR_Done,
    // Label 1059: @26718
    GIM_Try, /*On fail goto*//*Label 1060*/ 26741, // Rule ID 8056 //
      GIM_CheckFeatures, GIFBS_HasAVX512,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_s64,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::GR32RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::FR64XRegClassID,
      // (fp_to_sint:{ *:[i32] } FR64X:{ *:[f64] }:$src)  =>  (VCVTTSD2SIZrr:{ *:[i32] } FR64X:{ *:[f64] }:$src)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VCVTTSD2SIZrr,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 8056,
      GIR_Done,
    // Label 1060: @26741
    GIM_Reject,
    // Label 1047: @26742
    GIM_Try, /*On fail goto*//*Label 1061*/ 26765, // Rule ID 1337 //
      GIM_CheckFeatures, GIFBS_UseAVX,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_s32,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::GR64RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::FR32RegClassID,
      // (fp_to_sint:{ *:[i64] } FR32:{ *:[f32] }:$src)  =>  (VCVTTSS2SI64rr:{ *:[i64] } FR32:{ *:[f32] }:$src)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VCVTTSS2SI64rr,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 1337,
      GIR_Done,
    // Label 1061: @26765
    GIM_Try, /*On fail goto*//*Label 1062*/ 26788, // Rule ID 1341 //
      GIM_CheckFeatures, GIFBS_UseAVX,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_s64,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::GR64RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::FR64RegClassID,
      // (fp_to_sint:{ *:[i64] } FR64:{ *:[f64] }:$src)  =>  (VCVTTSD2SI64rr:{ *:[i64] } FR64:{ *:[f64] }:$src)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VCVTTSD2SI64rr,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 1341,
      GIR_Done,
    // Label 1062: @26788
    GIM_Try, /*On fail goto*//*Label 1063*/ 26811, // Rule ID 1345 //
      GIM_CheckFeatures, GIFBS_UseSSE1,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_s32,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::GR64RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::FR32RegClassID,
      // (fp_to_sint:{ *:[i64] } FR32:{ *:[f32] }:$src)  =>  (CVTTSS2SI64rr:{ *:[i64] } FR32:{ *:[f32] }:$src)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::CVTTSS2SI64rr,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 1345,
      GIR_Done,
    // Label 1063: @26811
    GIM_Try, /*On fail goto*//*Label 1064*/ 26834, // Rule ID 1349 //
      GIM_CheckFeatures, GIFBS_UseSSE2,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_s64,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::GR64RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::FR64RegClassID,
      // (fp_to_sint:{ *:[i64] } FR64:{ *:[f64] }:$src)  =>  (CVTTSD2SI64rr:{ *:[i64] } FR64:{ *:[f64] }:$src)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::CVTTSD2SI64rr,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 1349,
      GIR_Done,
    // Label 1064: @26834
    GIM_Try, /*On fail goto*//*Label 1065*/ 26857, // Rule ID 8051 //
      GIM_CheckFeatures, GIFBS_HasAVX512,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_s32,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::GR64RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::FR32XRegClassID,
      // (fp_to_sint:{ *:[i64] } FR32X:{ *:[f32] }:$src)  =>  (VCVTTSS2SI64Zrr:{ *:[i64] } FR32X:{ *:[f32] }:$src)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VCVTTSS2SI64Zrr,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 8051,
      GIR_Done,
    // Label 1065: @26857
    GIM_Try, /*On fail goto*//*Label 1066*/ 26880, // Rule ID 8061 //
      GIM_CheckFeatures, GIFBS_HasAVX512,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_s64,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::GR64RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::FR64XRegClassID,
      // (fp_to_sint:{ *:[i64] } FR64X:{ *:[f64] }:$src)  =>  (VCVTTSD2SI64Zrr:{ *:[i64] } FR64X:{ *:[f64] }:$src)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VCVTTSD2SI64Zrr,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 8061,
      GIR_Done,
    // Label 1066: @26880
    GIM_Reject,
    // Label 1048: @26881
    GIM_Try, /*On fail goto*//*Label 1067*/ 26904, // Rule ID 14856 //
      GIM_CheckFeatures, GIFBS_HasDQI_HasVLX,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v2s64,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128XRegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR128XRegClassID,
      // (fp_to_sint:{ *:[v2i64] } VR128X:{ *:[v2f64] }:$src)  =>  (VCVTTPD2QQZ128rr:{ *:[v2i64] } VR128X:{ *:[v2f64] }:$src)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VCVTTPD2QQZ128rr,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 14856,
      GIR_Done,
    // Label 1067: @26904
    GIM_Reject,
    // Label 1049: @26905
    GIM_Try, /*On fail goto*//*Label 1068*/ 26928, // Rule ID 12467 //
      GIM_CheckFeatures, GIFBS_HasAVX_NoVLX,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v4s32,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR128RegClassID,
      // (fp_to_sint:{ *:[v4i32] } VR128:{ *:[v4f32] }:$src)  =>  (VCVTTPS2DQrr:{ *:[v4i32] } VR128:{ *:[v4f32] }:$src)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VCVTTPS2DQrr,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 12467,
      GIR_Done,
    // Label 1068: @26928
    GIM_Try, /*On fail goto*//*Label 1069*/ 26951, // Rule ID 12471 //
      GIM_CheckFeatures, GIFBS_UseSSE2,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v4s32,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR128RegClassID,
      // (fp_to_sint:{ *:[v4i32] } VR128:{ *:[v4f32] }:$src)  =>  (CVTTPS2DQrr:{ *:[v4i32] } VR128:{ *:[v4f32] }:$src)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::CVTTPS2DQrr,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 12471,
      GIR_Done,
    // Label 1069: @26951
    GIM_Try, /*On fail goto*//*Label 1070*/ 26974, // Rule ID 12473 //
      GIM_CheckFeatures, GIFBS_HasAVX_NoVLX,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v4s64,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR256RegClassID,
      // (fp_to_sint:{ *:[v4i32] } VR256:{ *:[v4f64] }:$src)  =>  (VCVTTPD2DQYrr:{ *:[v4i32] } VR256:{ *:[v4f64] }:$src)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VCVTTPD2DQYrr,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 12473,
      GIR_Done,
    // Label 1070: @26974
    GIM_Try, /*On fail goto*//*Label 1071*/ 26997, // Rule ID 14832 //
      GIM_CheckFeatures, GIFBS_HasVLX,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v4s32,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128XRegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR128XRegClassID,
      // (fp_to_sint:{ *:[v4i32] } VR128X:{ *:[v4f32] }:$src)  =>  (VCVTTPS2DQZ128rr:{ *:[v4i32] } VR128X:{ *:[v4f32] }:$src)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VCVTTPS2DQZ128rr,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 14832,
      GIR_Done,
    // Label 1071: @26997
    GIM_Try, /*On fail goto*//*Label 1072*/ 27020, // Rule ID 14840 //
      GIM_CheckFeatures, GIFBS_HasVLX,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v4s64,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128XRegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR256XRegClassID,
      // (fp_to_sint:{ *:[v4i32] } VR256X:{ *:[v4f64] }:$src)  =>  (VCVTTPD2DQZ256rr:{ *:[v4i32] } VR256X:{ *:[v4f64] }:$src)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VCVTTPD2DQZ256rr,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 14840,
      GIR_Done,
    // Label 1072: @27020
    GIM_Reject,
    // Label 1050: @27021
    GIM_Try, /*On fail goto*//*Label 1073*/ 27044, // Rule ID 14852 //
      GIM_CheckFeatures, GIFBS_HasDQI_HasVLX,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v4s32,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR256XRegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR128XRegClassID,
      // (fp_to_sint:{ *:[v4i64] } VR128X:{ *:[v4f32] }:$src)  =>  (VCVTTPS2QQZ256rr:{ *:[v4i64] } VR128X:{ *:[v4f32] }:$src)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VCVTTPS2QQZ256rr,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 14852,
      GIR_Done,
    // Label 1073: @27044
    GIM_Try, /*On fail goto*//*Label 1074*/ 27067, // Rule ID 14860 //
      GIM_CheckFeatures, GIFBS_HasDQI_HasVLX,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v4s64,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR256XRegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR256XRegClassID,
      // (fp_to_sint:{ *:[v4i64] } VR256X:{ *:[v4f64] }:$src)  =>  (VCVTTPD2QQZ256rr:{ *:[v4i64] } VR256X:{ *:[v4f64] }:$src)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VCVTTPD2QQZ256rr,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 14860,
      GIR_Done,
    // Label 1074: @27067
    GIM_Reject,
    // Label 1051: @27068
    GIM_Try, /*On fail goto*//*Label 1075*/ 27091, // Rule ID 12469 //
      GIM_CheckFeatures, GIFBS_HasAVX_NoVLX,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v8s32,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR256RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR256RegClassID,
      // (fp_to_sint:{ *:[v8i32] } VR256:{ *:[v8f32] }:$src)  =>  (VCVTTPS2DQYrr:{ *:[v8i32] } VR256:{ *:[v8f32] }:$src)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VCVTTPS2DQYrr,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 12469,
      GIR_Done,
    // Label 1075: @27091
    GIM_Try, /*On fail goto*//*Label 1076*/ 27114, // Rule ID 14828 //
      GIM_CheckFeatures, GIFBS_HasAVX512,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v8s64,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR256XRegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR512RegClassID,
      // (fp_to_sint:{ *:[v8i32] } VR512:{ *:[v8f64] }:$src)  =>  (VCVTTPD2DQZrr:{ *:[v8i32] } VR512:{ *:[v8f64] }:$src)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VCVTTPD2DQZrr,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 14828,
      GIR_Done,
    // Label 1076: @27114
    GIM_Try, /*On fail goto*//*Label 1077*/ 27137, // Rule ID 14836 //
      GIM_CheckFeatures, GIFBS_HasVLX,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v8s32,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR256XRegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR256XRegClassID,
      // (fp_to_sint:{ *:[v8i32] } VR256X:{ *:[v8f32] }:$src)  =>  (VCVTTPS2DQZ256rr:{ *:[v8i32] } VR256X:{ *:[v8f32] }:$src)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VCVTTPS2DQZ256rr,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 14836,
      GIR_Done,
    // Label 1077: @27137
    GIM_Reject,
    // Label 1052: @27138
    GIM_Try, /*On fail goto*//*Label 1078*/ 27161, // Rule ID 14844 //
      GIM_CheckFeatures, GIFBS_HasDQI,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v8s32,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR512RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR256XRegClassID,
      // (fp_to_sint:{ *:[v8i64] } VR256X:{ *:[v8f32] }:$src)  =>  (VCVTTPS2QQZrr:{ *:[v8i64] } VR256X:{ *:[v8f32] }:$src)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VCVTTPS2QQZrr,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 14844,
      GIR_Done,
    // Label 1078: @27161
    GIM_Try, /*On fail goto*//*Label 1079*/ 27184, // Rule ID 14848 //
      GIM_CheckFeatures, GIFBS_HasDQI,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v8s64,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR512RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR512RegClassID,
      // (fp_to_sint:{ *:[v8i64] } VR512:{ *:[v8f64] }:$src)  =>  (VCVTTPD2QQZrr:{ *:[v8i64] } VR512:{ *:[v8f64] }:$src)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VCVTTPD2QQZrr,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 14848,
      GIR_Done,
    // Label 1079: @27184
    GIM_Reject,
    // Label 1053: @27185
    GIM_Try, /*On fail goto*//*Label 1080*/ 27208, // Rule ID 14824 //
      GIM_CheckFeatures, GIFBS_HasAVX512,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v16s32,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR512RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR512RegClassID,
      // (fp_to_sint:{ *:[v16i32] } VR512:{ *:[v16f32] }:$src)  =>  (VCVTTPS2DQZrr:{ *:[v16i32] } VR512:{ *:[v16f32] }:$src)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VCVTTPS2DQZrr,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 14824,
      GIR_Done,
    // Label 1080: @27208
    GIM_Reject,
    // Label 1054: @27209
    GIM_Reject,
    // Label 26: @27210
    GIM_SwitchType, /*MI*/0, /*Op*/0, /*[*/3, 20, /*)*//*default:*//*Label 1089*/ 27563,
    /*GILLT_s32*//*Label 1081*/ 27233,
    /*GILLT_s64*//*Label 1082*/ 27280, 0, 0, 0,
    /*GILLT_v2s64*//*Label 1083*/ 27327, 0,
    /*GILLT_v4s32*//*Label 1084*/ 27351,
    /*GILLT_v4s64*//*Label 1085*/ 27398, 0, 0,
    /*GILLT_v8s32*//*Label 1086*/ 27445,
    /*GILLT_v8s64*//*Label 1087*/ 27492, 0, 0, 0,
    /*GILLT_v16s32*//*Label 1088*/ 27539,
    // Label 1081: @27233
    GIM_Try, /*On fail goto*//*Label 1090*/ 27256, // Rule ID 8066 //
      GIM_CheckFeatures, GIFBS_HasAVX512,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_s32,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::GR32RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::FR32XRegClassID,
      // (fp_to_uint:{ *:[i32] } FR32X:{ *:[f32] }:$src)  =>  (VCVTTSS2USIZrr:{ *:[i32] } FR32X:{ *:[f32] }:$src)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VCVTTSS2USIZrr,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 8066,
      GIR_Done,
    // Label 1090: @27256
    GIM_Try, /*On fail goto*//*Label 1091*/ 27279, // Rule ID 8076 //
      GIM_CheckFeatures, GIFBS_HasAVX512,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_s64,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::GR32RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::FR64XRegClassID,
      // (fp_to_uint:{ *:[i32] } FR64X:{ *:[f64] }:$src)  =>  (VCVTTSD2USIZrr:{ *:[i32] } FR64X:{ *:[f64] }:$src)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VCVTTSD2USIZrr,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 8076,
      GIR_Done,
    // Label 1091: @27279
    GIM_Reject,
    // Label 1082: @27280
    GIM_Try, /*On fail goto*//*Label 1092*/ 27303, // Rule ID 8071 //
      GIM_CheckFeatures, GIFBS_HasAVX512,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_s32,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::GR64RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::FR32XRegClassID,
      // (fp_to_uint:{ *:[i64] } FR32X:{ *:[f32] }:$src)  =>  (VCVTTSS2USI64Zrr:{ *:[i64] } FR32X:{ *:[f32] }:$src)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VCVTTSS2USI64Zrr,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 8071,
      GIR_Done,
    // Label 1092: @27303
    GIM_Try, /*On fail goto*//*Label 1093*/ 27326, // Rule ID 8081 //
      GIM_CheckFeatures, GIFBS_HasAVX512,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_s64,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::GR64RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::FR64XRegClassID,
      // (fp_to_uint:{ *:[i64] } FR64X:{ *:[f64] }:$src)  =>  (VCVTTSD2USI64Zrr:{ *:[i64] } FR64X:{ *:[f64] }:$src)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VCVTTSD2USI64Zrr,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 8081,
      GIR_Done,
    // Label 1093: @27326
    GIM_Reject,
    // Label 1083: @27327
    GIM_Try, /*On fail goto*//*Label 1094*/ 27350, // Rule ID 14858 //
      GIM_CheckFeatures, GIFBS_HasDQI_HasVLX,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v2s64,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128XRegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR128XRegClassID,
      // (fp_to_uint:{ *:[v2i64] } VR128X:{ *:[v2f64] }:$src)  =>  (VCVTTPD2UQQZ128rr:{ *:[v2i64] } VR128X:{ *:[v2f64] }:$src)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VCVTTPD2UQQZ128rr,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 14858,
      GIR_Done,
    // Label 1094: @27350
    GIM_Reject,
    // Label 1084: @27351
    GIM_Try, /*On fail goto*//*Label 1095*/ 27374, // Rule ID 14834 //
      GIM_CheckFeatures, GIFBS_HasVLX,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v4s32,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128XRegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR128XRegClassID,
      // (fp_to_uint:{ *:[v4i32] } VR128X:{ *:[v4f32] }:$src)  =>  (VCVTTPS2UDQZ128rr:{ *:[v4i32] } VR128X:{ *:[v4f32] }:$src)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VCVTTPS2UDQZ128rr,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 14834,
      GIR_Done,
    // Label 1095: @27374
    GIM_Try, /*On fail goto*//*Label 1096*/ 27397, // Rule ID 14842 //
      GIM_CheckFeatures, GIFBS_HasVLX,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v4s64,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128XRegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR256XRegClassID,
      // (fp_to_uint:{ *:[v4i32] } VR256X:{ *:[v4f64] }:$src)  =>  (VCVTTPD2UDQZ256rr:{ *:[v4i32] } VR256X:{ *:[v4f64] }:$src)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VCVTTPD2UDQZ256rr,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 14842,
      GIR_Done,
    // Label 1096: @27397
    GIM_Reject,
    // Label 1085: @27398
    GIM_Try, /*On fail goto*//*Label 1097*/ 27421, // Rule ID 14854 //
      GIM_CheckFeatures, GIFBS_HasDQI_HasVLX,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v4s32,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR256XRegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR128XRegClassID,
      // (fp_to_uint:{ *:[v4i64] } VR128X:{ *:[v4f32] }:$src)  =>  (VCVTTPS2UQQZ256rr:{ *:[v4i64] } VR128X:{ *:[v4f32] }:$src)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VCVTTPS2UQQZ256rr,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 14854,
      GIR_Done,
    // Label 1097: @27421
    GIM_Try, /*On fail goto*//*Label 1098*/ 27444, // Rule ID 14862 //
      GIM_CheckFeatures, GIFBS_HasDQI_HasVLX,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v4s64,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR256XRegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR256XRegClassID,
      // (fp_to_uint:{ *:[v4i64] } VR256X:{ *:[v4f64] }:$src)  =>  (VCVTTPD2UQQZ256rr:{ *:[v4i64] } VR256X:{ *:[v4f64] }:$src)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VCVTTPD2UQQZ256rr,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 14862,
      GIR_Done,
    // Label 1098: @27444
    GIM_Reject,
    // Label 1086: @27445
    GIM_Try, /*On fail goto*//*Label 1099*/ 27468, // Rule ID 14830 //
      GIM_CheckFeatures, GIFBS_HasAVX512,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v8s64,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR256XRegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR512RegClassID,
      // (fp_to_uint:{ *:[v8i32] } VR512:{ *:[v8f64] }:$src)  =>  (VCVTTPD2UDQZrr:{ *:[v8i32] } VR512:{ *:[v8f64] }:$src)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VCVTTPD2UDQZrr,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 14830,
      GIR_Done,
    // Label 1099: @27468
    GIM_Try, /*On fail goto*//*Label 1100*/ 27491, // Rule ID 14838 //
      GIM_CheckFeatures, GIFBS_HasVLX,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v8s32,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR256XRegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR256XRegClassID,
      // (fp_to_uint:{ *:[v8i32] } VR256X:{ *:[v8f32] }:$src)  =>  (VCVTTPS2UDQZ256rr:{ *:[v8i32] } VR256X:{ *:[v8f32] }:$src)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VCVTTPS2UDQZ256rr,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 14838,
      GIR_Done,
    // Label 1100: @27491
    GIM_Reject,
    // Label 1087: @27492
    GIM_Try, /*On fail goto*//*Label 1101*/ 27515, // Rule ID 14846 //
      GIM_CheckFeatures, GIFBS_HasDQI,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v8s32,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR512RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR256XRegClassID,
      // (fp_to_uint:{ *:[v8i64] } VR256X:{ *:[v8f32] }:$src)  =>  (VCVTTPS2UQQZrr:{ *:[v8i64] } VR256X:{ *:[v8f32] }:$src)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VCVTTPS2UQQZrr,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 14846,
      GIR_Done,
    // Label 1101: @27515
    GIM_Try, /*On fail goto*//*Label 1102*/ 27538, // Rule ID 14850 //
      GIM_CheckFeatures, GIFBS_HasDQI,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v8s64,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR512RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR512RegClassID,
      // (fp_to_uint:{ *:[v8i64] } VR512:{ *:[v8f64] }:$src)  =>  (VCVTTPD2UQQZrr:{ *:[v8i64] } VR512:{ *:[v8f64] }:$src)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VCVTTPD2UQQZrr,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 14850,
      GIR_Done,
    // Label 1102: @27538
    GIM_Reject,
    // Label 1088: @27539
    GIM_Try, /*On fail goto*//*Label 1103*/ 27562, // Rule ID 14826 //
      GIM_CheckFeatures, GIFBS_HasAVX512,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v16s32,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR512RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR512RegClassID,
      // (fp_to_uint:{ *:[v16i32] } VR512:{ *:[v16f32] }:$src)  =>  (VCVTTPS2UDQZrr:{ *:[v16i32] } VR512:{ *:[v16f32] }:$src)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VCVTTPS2UDQZrr,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 14826,
      GIR_Done,
    // Label 1103: @27562
    GIM_Reject,
    // Label 1089: @27563
    GIM_Reject,
    // Label 27: @27564
    GIM_SwitchType, /*MI*/0, /*Op*/0, /*[*/3, 20, /*)*//*default:*//*Label 1112*/ 28393,
    /*GILLT_s32*//*Label 1104*/ 27587,
    /*GILLT_s64*//*Label 1105*/ 27826, 0, 0, 0,
    /*GILLT_v2s64*//*Label 1106*/ 28065, 0,
    /*GILLT_v4s32*//*Label 1107*/ 28089,
    /*GILLT_v4s64*//*Label 1108*/ 28182, 0, 0,
    /*GILLT_v8s32*//*Label 1109*/ 28252,
    /*GILLT_v8s64*//*Label 1110*/ 28322, 0, 0, 0,
    /*GILLT_v16s32*//*Label 1111*/ 28369,
    // Label 1104: @27587
    GIM_Try, /*On fail goto*//*Label 1113*/ 27610, // Rule ID 1351 //
      GIM_CheckFeatures, GIFBS_UseSSE1,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_s32,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::FR32RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::GR32RegClassID,
      // (sint_to_fp:{ *:[f32] } GR32:{ *:[i32] }:$src)  =>  (CVTSI2SSrr:{ *:[f32] } GR32:{ *:[i32] }:$src)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::CVTSI2SSrr,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 1351,
      GIR_Done,
    // Label 1113: @27610
    GIM_Try, /*On fail goto*//*Label 1114*/ 27633, // Rule ID 1353 //
      GIM_CheckFeatures, GIFBS_UseSSE1,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_s64,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::FR32RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::GR64RegClassID,
      // (sint_to_fp:{ *:[f32] } GR64:{ *:[i64] }:$src)  =>  (CVTSI642SSrr:{ *:[f32] } GR64:{ *:[i64] }:$src)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::CVTSI642SSrr,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 1353,
      GIR_Done,
    // Label 1114: @27633
    GIM_Try, /*On fail goto*//*Label 1115*/ 27681, // Rule ID 12436 //
      GIM_CheckFeatures, GIFBS_UseAVX,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_s32,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::FR32RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::GR32RegClassID,
      // (sint_to_fp:{ *:[f32] } GR32:{ *:[i32] }:$src)  =>  (VCVTSI2SSrr:{ *:[f32] } (IMPLICIT_DEF:{ *:[f32] }), GR32:{ *:[i32] }:$src)
      GIR_MakeTempReg, /*TempRegID*/0, /*TypeID*/GILLT_s32,
      GIR_BuildMI, /*InsnID*/1, /*Opcode*/TargetOpcode::IMPLICIT_DEF,
      GIR_AddTempRegister, /*InsnID*/1, /*TempRegID*/0, /*TempRegFlags*/RegState::Define,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/1,
      GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::VCVTSI2SSrr,
      GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
      GIR_AddTempRegister, /*InsnID*/0, /*TempRegID*/0, /*TempRegFlags*/0,
      GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // src
      GIR_EraseFromParent, /*InsnID*/0,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 12436,
      GIR_Done,
    // Label 1115: @27681
    GIM_Try, /*On fail goto*//*Label 1116*/ 27729, // Rule ID 12437 //
      GIM_CheckFeatures, GIFBS_UseAVX,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_s64,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::FR32RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::GR64RegClassID,
      // (sint_to_fp:{ *:[f32] } GR64:{ *:[i64] }:$src)  =>  (VCVTSI642SSrr:{ *:[f32] } (IMPLICIT_DEF:{ *:[f32] }), GR64:{ *:[i64] }:$src)
      GIR_MakeTempReg, /*TempRegID*/0, /*TypeID*/GILLT_s32,
      GIR_BuildMI, /*InsnID*/1, /*Opcode*/TargetOpcode::IMPLICIT_DEF,
      GIR_AddTempRegister, /*InsnID*/1, /*TempRegID*/0, /*TempRegFlags*/RegState::Define,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/1,
      GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::VCVTSI642SSrr,
      GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
      GIR_AddTempRegister, /*InsnID*/0, /*TempRegID*/0, /*TempRegFlags*/0,
      GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // src
      GIR_EraseFromParent, /*InsnID*/0,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 12437,
      GIR_Done,
    // Label 1116: @27729
    GIM_Try, /*On fail goto*//*Label 1117*/ 27777, // Rule ID 14784 //
      GIM_CheckFeatures, GIFBS_HasAVX512,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_s32,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::FR32XRegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::GR32RegClassID,
      // (sint_to_fp:{ *:[f32] } GR32:{ *:[i32] }:$src)  =>  (VCVTSI2SSZrr:{ *:[f32] } (IMPLICIT_DEF:{ *:[f32] }), GR32:{ *:[i32] }:$src)
      GIR_MakeTempReg, /*TempRegID*/0, /*TypeID*/GILLT_s32,
      GIR_BuildMI, /*InsnID*/1, /*Opcode*/TargetOpcode::IMPLICIT_DEF,
      GIR_AddTempRegister, /*InsnID*/1, /*TempRegID*/0, /*TempRegFlags*/RegState::Define,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/1,
      GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::VCVTSI2SSZrr,
      GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
      GIR_AddTempRegister, /*InsnID*/0, /*TempRegID*/0, /*TempRegFlags*/0,
      GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // src
      GIR_EraseFromParent, /*InsnID*/0,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 14784,
      GIR_Done,
    // Label 1117: @27777
    GIM_Try, /*On fail goto*//*Label 1118*/ 27825, // Rule ID 14785 //
      GIM_CheckFeatures, GIFBS_HasAVX512,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_s64,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::FR32XRegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::GR64RegClassID,
      // (sint_to_fp:{ *:[f32] } GR64:{ *:[i64] }:$src)  =>  (VCVTSI642SSZrr:{ *:[f32] } (IMPLICIT_DEF:{ *:[f32] }), GR64:{ *:[i64] }:$src)
      GIR_MakeTempReg, /*TempRegID*/0, /*TypeID*/GILLT_s32,
      GIR_BuildMI, /*InsnID*/1, /*Opcode*/TargetOpcode::IMPLICIT_DEF,
      GIR_AddTempRegister, /*InsnID*/1, /*TempRegID*/0, /*TempRegFlags*/RegState::Define,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/1,
      GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::VCVTSI642SSZrr,
      GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
      GIR_AddTempRegister, /*InsnID*/0, /*TempRegID*/0, /*TempRegFlags*/0,
      GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // src
      GIR_EraseFromParent, /*InsnID*/0,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 14785,
      GIR_Done,
    // Label 1118: @27825
    GIM_Reject,
    // Label 1105: @27826
    GIM_Try, /*On fail goto*//*Label 1119*/ 27849, // Rule ID 1355 //
      GIM_CheckFeatures, GIFBS_UseSSE2,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_s32,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::FR64RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::GR32RegClassID,
      // (sint_to_fp:{ *:[f64] } GR32:{ *:[i32] }:$src)  =>  (CVTSI2SDrr:{ *:[f64] } GR32:{ *:[i32] }:$src)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::CVTSI2SDrr,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 1355,
      GIR_Done,
    // Label 1119: @27849
    GIM_Try, /*On fail goto*//*Label 1120*/ 27872, // Rule ID 1357 //
      GIM_CheckFeatures, GIFBS_UseSSE2,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_s64,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::FR64RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::GR64RegClassID,
      // (sint_to_fp:{ *:[f64] } GR64:{ *:[i64] }:$src)  =>  (CVTSI642SDrr:{ *:[f64] } GR64:{ *:[i64] }:$src)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::CVTSI642SDrr,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 1357,
      GIR_Done,
    // Label 1120: @27872
    GIM_Try, /*On fail goto*//*Label 1121*/ 27920, // Rule ID 12438 //
      GIM_CheckFeatures, GIFBS_UseAVX,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_s32,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::FR64RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::GR32RegClassID,
      // (sint_to_fp:{ *:[f64] } GR32:{ *:[i32] }:$src)  =>  (VCVTSI2SDrr:{ *:[f64] } (IMPLICIT_DEF:{ *:[f64] }), GR32:{ *:[i32] }:$src)
      GIR_MakeTempReg, /*TempRegID*/0, /*TypeID*/GILLT_s64,
      GIR_BuildMI, /*InsnID*/1, /*Opcode*/TargetOpcode::IMPLICIT_DEF,
      GIR_AddTempRegister, /*InsnID*/1, /*TempRegID*/0, /*TempRegFlags*/RegState::Define,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/1,
      GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::VCVTSI2SDrr,
      GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
      GIR_AddTempRegister, /*InsnID*/0, /*TempRegID*/0, /*TempRegFlags*/0,
      GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // src
      GIR_EraseFromParent, /*InsnID*/0,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 12438,
      GIR_Done,
    // Label 1121: @27920
    GIM_Try, /*On fail goto*//*Label 1122*/ 27968, // Rule ID 12439 //
      GIM_CheckFeatures, GIFBS_UseAVX,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_s64,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::FR64RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::GR64RegClassID,
      // (sint_to_fp:{ *:[f64] } GR64:{ *:[i64] }:$src)  =>  (VCVTSI642SDrr:{ *:[f64] } (IMPLICIT_DEF:{ *:[f64] }), GR64:{ *:[i64] }:$src)
      GIR_MakeTempReg, /*TempRegID*/0, /*TypeID*/GILLT_s64,
      GIR_BuildMI, /*InsnID*/1, /*Opcode*/TargetOpcode::IMPLICIT_DEF,
      GIR_AddTempRegister, /*InsnID*/1, /*TempRegID*/0, /*TempRegFlags*/RegState::Define,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/1,
      GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::VCVTSI642SDrr,
      GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
      GIR_AddTempRegister, /*InsnID*/0, /*TempRegID*/0, /*TempRegFlags*/0,
      GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // src
      GIR_EraseFromParent, /*InsnID*/0,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 12439,
      GIR_Done,
    // Label 1122: @27968
    GIM_Try, /*On fail goto*//*Label 1123*/ 28016, // Rule ID 14786 //
      GIM_CheckFeatures, GIFBS_HasAVX512,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_s32,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::FR64XRegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::GR32RegClassID,
      // (sint_to_fp:{ *:[f64] } GR32:{ *:[i32] }:$src)  =>  (VCVTSI2SDZrr:{ *:[f64] } (IMPLICIT_DEF:{ *:[f64] }), GR32:{ *:[i32] }:$src)
      GIR_MakeTempReg, /*TempRegID*/0, /*TypeID*/GILLT_s64,
      GIR_BuildMI, /*InsnID*/1, /*Opcode*/TargetOpcode::IMPLICIT_DEF,
      GIR_AddTempRegister, /*InsnID*/1, /*TempRegID*/0, /*TempRegFlags*/RegState::Define,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/1,
      GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::VCVTSI2SDZrr,
      GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
      GIR_AddTempRegister, /*InsnID*/0, /*TempRegID*/0, /*TempRegFlags*/0,
      GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // src
      GIR_EraseFromParent, /*InsnID*/0,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 14786,
      GIR_Done,
    // Label 1123: @28016
    GIM_Try, /*On fail goto*//*Label 1124*/ 28064, // Rule ID 14787 //
      GIM_CheckFeatures, GIFBS_HasAVX512,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_s64,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::FR64XRegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::GR64RegClassID,
      // (sint_to_fp:{ *:[f64] } GR64:{ *:[i64] }:$src)  =>  (VCVTSI642SDZrr:{ *:[f64] } (IMPLICIT_DEF:{ *:[f64] }), GR64:{ *:[i64] }:$src)
      GIR_MakeTempReg, /*TempRegID*/0, /*TypeID*/GILLT_s64,
      GIR_BuildMI, /*InsnID*/1, /*Opcode*/TargetOpcode::IMPLICIT_DEF,
      GIR_AddTempRegister, /*InsnID*/1, /*TempRegID*/0, /*TempRegFlags*/RegState::Define,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/1,
      GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::VCVTSI642SDZrr,
      GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
      GIR_AddTempRegister, /*InsnID*/0, /*TempRegID*/0, /*TempRegFlags*/0,
      GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // src
      GIR_EraseFromParent, /*InsnID*/0,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 14787,
      GIR_Done,
    // Label 1124: @28064
    GIM_Reject,
    // Label 1106: @28065
    GIM_Try, /*On fail goto*//*Label 1125*/ 28088, // Rule ID 8770 //
      GIM_CheckFeatures, GIFBS_HasDQI_HasVLX,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v2s64,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128XRegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR128XRegClassID,
      // (sint_to_fp:{ *:[v2f64] } VR128X:{ *:[v2i64] }:$src)  =>  (VCVTQQ2PDZ128rr:{ *:[v2f64] } VR128X:{ *:[v2i64] }:$src)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VCVTQQ2PDZ128rr,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 8770,
      GIR_Done,
    // Label 1125: @28088
    GIM_Reject,
    // Label 1107: @28089
    GIM_Try, /*On fail goto*//*Label 1126*/ 28112, // Rule ID 1391 //
      GIM_CheckFeatures, GIFBS_HasAVX_NoVLX,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v4s32,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR128RegClassID,
      // (sint_to_fp:{ *:[v4f32] } VR128:{ *:[v4i32] }:$src)  =>  (VCVTDQ2PSrr:{ *:[v4f32] } VR128:{ *:[v4i32] }:$src)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VCVTDQ2PSrr,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 1391,
      GIR_Done,
    // Label 1126: @28112
    GIM_Try, /*On fail goto*//*Label 1127*/ 28135, // Rule ID 1395 //
      GIM_CheckFeatures, GIFBS_UseSSE2,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v4s32,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR128RegClassID,
      // (sint_to_fp:{ *:[v4f32] } VR128:{ *:[v4i32] }:$src)  =>  (CVTDQ2PSrr:{ *:[v4f32] } VR128:{ *:[v4i32] }:$src)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::CVTDQ2PSrr,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 1395,
      GIR_Done,
    // Label 1127: @28135
    GIM_Try, /*On fail goto*//*Label 1128*/ 28158, // Rule ID 8203 //
      GIM_CheckFeatures, GIFBS_HasVLX,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v4s32,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128XRegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR128XRegClassID,
      // (sint_to_fp:{ *:[v4f32] } VR128X:{ *:[v4i32] }:$src)  =>  (VCVTDQ2PSZ128rr:{ *:[v4f32] } VR128X:{ *:[v4i32] }:$src)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VCVTDQ2PSZ128rr,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 8203,
      GIR_Done,
    // Label 1128: @28158
    GIM_Try, /*On fail goto*//*Label 1129*/ 28181, // Rule ID 8839 //
      GIM_CheckFeatures, GIFBS_HasDQI_HasVLX,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v4s64,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128XRegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR256XRegClassID,
      // (sint_to_fp:{ *:[v4f32] } VR256X:{ *:[v4i64] }:$src)  =>  (VCVTQQ2PSZ256rr:{ *:[v4f32] } VR256X:{ *:[v4i64] }:$src)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VCVTQQ2PSZ256rr,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 8839,
      GIR_Done,
    // Label 1129: @28181
    GIM_Reject,
    // Label 1108: @28182
    GIM_Try, /*On fail goto*//*Label 1130*/ 28205, // Rule ID 1438 //
      GIM_CheckFeatures, GIFBS_HasAVX_NoVLX,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v4s32,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR256RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR128RegClassID,
      // (sint_to_fp:{ *:[v4f64] } VR128:{ *:[v4i32] }:$src)  =>  (VCVTDQ2PDYrr:{ *:[v4f64] } VR128:{ *:[v4i32] }:$src)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VCVTDQ2PDYrr,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 1438,
      GIR_Done,
    // Label 1130: @28205
    GIM_Try, /*On fail goto*//*Label 1131*/ 28228, // Rule ID 8182 //
      GIM_CheckFeatures, GIFBS_HasVLX,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v4s32,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR256XRegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR128XRegClassID,
      // (sint_to_fp:{ *:[v4f64] } VR128X:{ *:[v4i32] }:$src)  =>  (VCVTDQ2PDZ256rr:{ *:[v4f64] } VR128X:{ *:[v4i32] }:$src)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VCVTDQ2PDZ256rr,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 8182,
      GIR_Done,
    // Label 1131: @28228
    GIM_Try, /*On fail goto*//*Label 1132*/ 28251, // Rule ID 8779 //
      GIM_CheckFeatures, GIFBS_HasDQI_HasVLX,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v4s64,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR256XRegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR256XRegClassID,
      // (sint_to_fp:{ *:[v4f64] } VR256X:{ *:[v4i64] }:$src)  =>  (VCVTQQ2PDZ256rr:{ *:[v4f64] } VR256X:{ *:[v4i64] }:$src)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VCVTQQ2PDZ256rr,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 8779,
      GIR_Done,
    // Label 1132: @28251
    GIM_Reject,
    // Label 1109: @28252
    GIM_Try, /*On fail goto*//*Label 1133*/ 28275, // Rule ID 1393 //
      GIM_CheckFeatures, GIFBS_HasAVX_NoVLX,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v8s32,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR256RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR256RegClassID,
      // (sint_to_fp:{ *:[v8f32] } VR256:{ *:[v8i32] }:$src)  =>  (VCVTDQ2PSYrr:{ *:[v8f32] } VR256:{ *:[v8i32] }:$src)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VCVTDQ2PSYrr,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 1393,
      GIR_Done,
    // Label 1133: @28275
    GIM_Try, /*On fail goto*//*Label 1134*/ 28298, // Rule ID 8212 //
      GIM_CheckFeatures, GIFBS_HasVLX,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v8s32,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR256XRegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR256XRegClassID,
      // (sint_to_fp:{ *:[v8f32] } VR256X:{ *:[v8i32] }:$src)  =>  (VCVTDQ2PSZ256rr:{ *:[v8f32] } VR256X:{ *:[v8i32] }:$src)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VCVTDQ2PSZ256rr,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 8212,
      GIR_Done,
    // Label 1134: @28298
    GIM_Try, /*On fail goto*//*Label 1135*/ 28321, // Rule ID 8818 //
      GIM_CheckFeatures, GIFBS_HasDQI,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v8s64,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR256XRegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR512RegClassID,
      // (sint_to_fp:{ *:[v8f32] } VR512:{ *:[v8i64] }:$src)  =>  (VCVTQQ2PSZrr:{ *:[v8f32] } VR512:{ *:[v8i64] }:$src)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VCVTQQ2PSZrr,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 8818,
      GIR_Done,
    // Label 1135: @28321
    GIM_Reject,
    // Label 1110: @28322
    GIM_Try, /*On fail goto*//*Label 1136*/ 28345, // Rule ID 8164 //
      GIM_CheckFeatures, GIFBS_HasAVX512,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v8s32,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR512RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR256XRegClassID,
      // (sint_to_fp:{ *:[v8f64] } VR256X:{ *:[v8i32] }:$src)  =>  (VCVTDQ2PDZrr:{ *:[v8f64] } VR256X:{ *:[v8i32] }:$src)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VCVTDQ2PDZrr,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 8164,
      GIR_Done,
    // Label 1136: @28345
    GIM_Try, /*On fail goto*//*Label 1137*/ 28368, // Rule ID 8758 //
      GIM_CheckFeatures, GIFBS_HasDQI,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v8s64,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR512RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR512RegClassID,
      // (sint_to_fp:{ *:[v8f64] } VR512:{ *:[v8i64] }:$src)  =>  (VCVTQQ2PDZrr:{ *:[v8f64] } VR512:{ *:[v8i64] }:$src)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VCVTQQ2PDZrr,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 8758,
      GIR_Done,
    // Label 1137: @28368
    GIM_Reject,
    // Label 1111: @28369
    GIM_Try, /*On fail goto*//*Label 1138*/ 28392, // Rule ID 8191 //
      GIM_CheckFeatures, GIFBS_HasAVX512,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v16s32,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR512RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR512RegClassID,
      // (sint_to_fp:{ *:[v16f32] } VR512:{ *:[v16i32] }:$src)  =>  (VCVTDQ2PSZrr:{ *:[v16f32] } VR512:{ *:[v16i32] }:$src)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VCVTDQ2PSZrr,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 8191,
      GIR_Done,
    // Label 1138: @28392
    GIM_Reject,
    // Label 1112: @28393
    GIM_Reject,
    // Label 28: @28394
    GIM_SwitchType, /*MI*/0, /*Op*/0, /*[*/3, 20, /*)*//*default:*//*Label 1147*/ 28847,
    /*GILLT_s32*//*Label 1139*/ 28417,
    /*GILLT_s64*//*Label 1140*/ 28514, 0, 0, 0,
    /*GILLT_v2s64*//*Label 1141*/ 28611, 0,
    /*GILLT_v4s32*//*Label 1142*/ 28635,
    /*GILLT_v4s64*//*Label 1143*/ 28682, 0, 0,
    /*GILLT_v8s32*//*Label 1144*/ 28729,
    /*GILLT_v8s64*//*Label 1145*/ 28776, 0, 0, 0,
    /*GILLT_v16s32*//*Label 1146*/ 28823,
    // Label 1139: @28417
    GIM_Try, /*On fail goto*//*Label 1148*/ 28465, // Rule ID 14792 //
      GIM_CheckFeatures, GIFBS_HasAVX512,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_s32,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::FR32XRegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::GR32RegClassID,
      // (uint_to_fp:{ *:[f32] } GR32:{ *:[i32] }:$src)  =>  (VCVTUSI2SSZrr:{ *:[f32] } (IMPLICIT_DEF:{ *:[f32] }), GR32:{ *:[i32] }:$src)
      GIR_MakeTempReg, /*TempRegID*/0, /*TypeID*/GILLT_s32,
      GIR_BuildMI, /*InsnID*/1, /*Opcode*/TargetOpcode::IMPLICIT_DEF,
      GIR_AddTempRegister, /*InsnID*/1, /*TempRegID*/0, /*TempRegFlags*/RegState::Define,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/1,
      GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::VCVTUSI2SSZrr,
      GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
      GIR_AddTempRegister, /*InsnID*/0, /*TempRegID*/0, /*TempRegFlags*/0,
      GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // src
      GIR_EraseFromParent, /*InsnID*/0,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 14792,
      GIR_Done,
    // Label 1148: @28465
    GIM_Try, /*On fail goto*//*Label 1149*/ 28513, // Rule ID 14793 //
      GIM_CheckFeatures, GIFBS_HasAVX512,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_s64,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::FR32XRegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::GR64RegClassID,
      // (uint_to_fp:{ *:[f32] } GR64:{ *:[i64] }:$src)  =>  (VCVTUSI642SSZrr:{ *:[f32] } (IMPLICIT_DEF:{ *:[f32] }), GR64:{ *:[i64] }:$src)
      GIR_MakeTempReg, /*TempRegID*/0, /*TypeID*/GILLT_s32,
      GIR_BuildMI, /*InsnID*/1, /*Opcode*/TargetOpcode::IMPLICIT_DEF,
      GIR_AddTempRegister, /*InsnID*/1, /*TempRegID*/0, /*TempRegFlags*/RegState::Define,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/1,
      GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::VCVTUSI642SSZrr,
      GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
      GIR_AddTempRegister, /*InsnID*/0, /*TempRegID*/0, /*TempRegFlags*/0,
      GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // src
      GIR_EraseFromParent, /*InsnID*/0,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 14793,
      GIR_Done,
    // Label 1149: @28513
    GIM_Reject,
    // Label 1140: @28514
    GIM_Try, /*On fail goto*//*Label 1150*/ 28562, // Rule ID 14794 //
      GIM_CheckFeatures, GIFBS_HasAVX512,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_s32,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::FR64XRegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::GR32RegClassID,
      // (uint_to_fp:{ *:[f64] } GR32:{ *:[i32] }:$src)  =>  (VCVTUSI2SDZrr:{ *:[f64] } (IMPLICIT_DEF:{ *:[f64] }), GR32:{ *:[i32] }:$src)
      GIR_MakeTempReg, /*TempRegID*/0, /*TypeID*/GILLT_s64,
      GIR_BuildMI, /*InsnID*/1, /*Opcode*/TargetOpcode::IMPLICIT_DEF,
      GIR_AddTempRegister, /*InsnID*/1, /*TempRegID*/0, /*TempRegFlags*/RegState::Define,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/1,
      GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::VCVTUSI2SDZrr,
      GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
      GIR_AddTempRegister, /*InsnID*/0, /*TempRegID*/0, /*TempRegFlags*/0,
      GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // src
      GIR_EraseFromParent, /*InsnID*/0,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 14794,
      GIR_Done,
    // Label 1150: @28562
    GIM_Try, /*On fail goto*//*Label 1151*/ 28610, // Rule ID 14795 //
      GIM_CheckFeatures, GIFBS_HasAVX512,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_s64,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::FR64XRegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::GR64RegClassID,
      // (uint_to_fp:{ *:[f64] } GR64:{ *:[i64] }:$src)  =>  (VCVTUSI642SDZrr:{ *:[f64] } (IMPLICIT_DEF:{ *:[f64] }), GR64:{ *:[i64] }:$src)
      GIR_MakeTempReg, /*TempRegID*/0, /*TypeID*/GILLT_s64,
      GIR_BuildMI, /*InsnID*/1, /*Opcode*/TargetOpcode::IMPLICIT_DEF,
      GIR_AddTempRegister, /*InsnID*/1, /*TempRegID*/0, /*TempRegFlags*/RegState::Define,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/1,
      GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::VCVTUSI642SDZrr,
      GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
      GIR_AddTempRegister, /*InsnID*/0, /*TempRegID*/0, /*TempRegFlags*/0,
      GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // src
      GIR_EraseFromParent, /*InsnID*/0,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 14795,
      GIR_Done,
    // Label 1151: @28610
    GIM_Reject,
    // Label 1141: @28611
    GIM_Try, /*On fail goto*//*Label 1152*/ 28634, // Rule ID 8800 //
      GIM_CheckFeatures, GIFBS_HasDQI_HasVLX,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v2s64,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128XRegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR128XRegClassID,
      // (uint_to_fp:{ *:[v2f64] } VR128X:{ *:[v2i64] }:$src)  =>  (VCVTUQQ2PDZ128rr:{ *:[v2f64] } VR128X:{ *:[v2i64] }:$src)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VCVTUQQ2PDZ128rr,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 8800,
      GIR_Done,
    // Label 1152: @28634
    GIM_Reject,
    // Label 1142: @28635
    GIM_Try, /*On fail goto*//*Label 1153*/ 28658, // Rule ID 8380 //
      GIM_CheckFeatures, GIFBS_HasVLX,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v4s32,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128XRegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR128XRegClassID,
      // (uint_to_fp:{ *:[v4f32] } VR128X:{ *:[v4i32] }:$src)  =>  (VCVTUDQ2PSZ128rr:{ *:[v4f32] } VR128X:{ *:[v4i32] }:$src)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VCVTUDQ2PSZ128rr,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 8380,
      GIR_Done,
    // Label 1153: @28658
    GIM_Try, /*On fail goto*//*Label 1154*/ 28681, // Rule ID 8869 //
      GIM_CheckFeatures, GIFBS_HasDQI_HasVLX,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v4s64,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128XRegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR256XRegClassID,
      // (uint_to_fp:{ *:[v4f32] } VR256X:{ *:[v4i64] }:$src)  =>  (VCVTUQQ2PSZ256rr:{ *:[v4f32] } VR256X:{ *:[v4i64] }:$src)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VCVTUQQ2PSZ256rr,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 8869,
      GIR_Done,
    // Label 1154: @28681
    GIM_Reject,
    // Label 1143: @28682
    GIM_Try, /*On fail goto*//*Label 1155*/ 28705, // Rule ID 8359 //
      GIM_CheckFeatures, GIFBS_HasVLX,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v4s32,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR256XRegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR128XRegClassID,
      // (uint_to_fp:{ *:[v4f64] } VR128X:{ *:[v4i32] }:$src)  =>  (VCVTUDQ2PDZ256rr:{ *:[v4f64] } VR128X:{ *:[v4i32] }:$src)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VCVTUDQ2PDZ256rr,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 8359,
      GIR_Done,
    // Label 1155: @28705
    GIM_Try, /*On fail goto*//*Label 1156*/ 28728, // Rule ID 8809 //
      GIM_CheckFeatures, GIFBS_HasDQI_HasVLX,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v4s64,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR256XRegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR256XRegClassID,
      // (uint_to_fp:{ *:[v4f64] } VR256X:{ *:[v4i64] }:$src)  =>  (VCVTUQQ2PDZ256rr:{ *:[v4f64] } VR256X:{ *:[v4i64] }:$src)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VCVTUQQ2PDZ256rr,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 8809,
      GIR_Done,
    // Label 1156: @28728
    GIM_Reject,
    // Label 1144: @28729
    GIM_Try, /*On fail goto*//*Label 1157*/ 28752, // Rule ID 8389 //
      GIM_CheckFeatures, GIFBS_HasVLX,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v8s32,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR256XRegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR256XRegClassID,
      // (uint_to_fp:{ *:[v8f32] } VR256X:{ *:[v8i32] }:$src)  =>  (VCVTUDQ2PSZ256rr:{ *:[v8f32] } VR256X:{ *:[v8i32] }:$src)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VCVTUDQ2PSZ256rr,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 8389,
      GIR_Done,
    // Label 1157: @28752
    GIM_Try, /*On fail goto*//*Label 1158*/ 28775, // Rule ID 8848 //
      GIM_CheckFeatures, GIFBS_HasDQI,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v8s64,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR256XRegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR512RegClassID,
      // (uint_to_fp:{ *:[v8f32] } VR512:{ *:[v8i64] }:$src)  =>  (VCVTUQQ2PSZrr:{ *:[v8f32] } VR512:{ *:[v8i64] }:$src)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VCVTUQQ2PSZrr,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 8848,
      GIR_Done,
    // Label 1158: @28775
    GIM_Reject,
    // Label 1145: @28776
    GIM_Try, /*On fail goto*//*Label 1159*/ 28799, // Rule ID 8341 //
      GIM_CheckFeatures, GIFBS_HasAVX512,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v8s32,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR512RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR256XRegClassID,
      // (uint_to_fp:{ *:[v8f64] } VR256X:{ *:[v8i32] }:$src)  =>  (VCVTUDQ2PDZrr:{ *:[v8f64] } VR256X:{ *:[v8i32] }:$src)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VCVTUDQ2PDZrr,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 8341,
      GIR_Done,
    // Label 1159: @28799
    GIM_Try, /*On fail goto*//*Label 1160*/ 28822, // Rule ID 8788 //
      GIM_CheckFeatures, GIFBS_HasDQI,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v8s64,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR512RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR512RegClassID,
      // (uint_to_fp:{ *:[v8f64] } VR512:{ *:[v8i64] }:$src)  =>  (VCVTUQQ2PDZrr:{ *:[v8f64] } VR512:{ *:[v8i64] }:$src)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VCVTUQQ2PDZrr,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 8788,
      GIR_Done,
    // Label 1160: @28822
    GIM_Reject,
    // Label 1146: @28823
    GIM_Try, /*On fail goto*//*Label 1161*/ 28846, // Rule ID 8368 //
      GIM_CheckFeatures, GIFBS_HasAVX512,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v16s32,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR512RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR512RegClassID,
      // (uint_to_fp:{ *:[v16f32] } VR512:{ *:[v16i32] }:$src)  =>  (VCVTUDQ2PSZrr:{ *:[v16f32] } VR512:{ *:[v16i32] }:$src)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VCVTUDQ2PSZrr,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 8368,
      GIR_Done,
    // Label 1161: @28846
    GIM_Reject,
    // Label 1147: @28847
    GIM_Reject,
    // Label 29: @28848
    GIM_Try, /*On fail goto*//*Label 1162*/ 28860, // Rule ID 548 //
      // MIs[0] dst
      GIM_CheckIsMBB, /*MI*/0, /*Op*/0,
      // (br (bb:{ *:[Other] }):$dst)  =>  (JMP_1 (bb:{ *:[Other] }):$dst)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::JMP_1,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 548,
      GIR_Done,
    // Label 1162: @28860
    GIM_Reject,
    // Label 30: @28861
    GIM_SwitchType, /*MI*/0, /*Op*/0, /*[*/2, 5, /*)*//*default:*//*Label 1166*/ 28945,
    /*GILLT_s16*//*Label 1163*/ 28870,
    /*GILLT_s32*//*Label 1164*/ 28895,
    /*GILLT_s64*//*Label 1165*/ 28920,
    // Label 1163: @28870
    GIM_Try, /*On fail goto*//*Label 1167*/ 28894, // Rule ID 16201 //
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_s16,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::GR16RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::GR16RegClassID,
      // (cttz_zero_undef:{ *:[i16] } GR16:{ *:[i16] }:$src)  =>  (BSF16rr:{ *:[i16] }:{ *:[i32] } GR16:{ *:[i16] }:$src)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::BSF16rr,
      GIR_AddImplicitDef, /*InsnID*/0, X86::EFLAGS,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 16201,
      GIR_Done,
    // Label 1167: @28894
    GIM_Reject,
    // Label 1164: @28895
    GIM_Try, /*On fail goto*//*Label 1168*/ 28919, // Rule ID 16202 //
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_s32,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::GR32RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::GR32RegClassID,
      // (cttz_zero_undef:{ *:[i32] } GR32:{ *:[i32] }:$src)  =>  (BSF32rr:{ *:[i32] }:{ *:[i32] } GR32:{ *:[i32] }:$src)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::BSF32rr,
      GIR_AddImplicitDef, /*InsnID*/0, X86::EFLAGS,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 16202,
      GIR_Done,
    // Label 1168: @28919
    GIM_Reject,
    // Label 1165: @28920
    GIM_Try, /*On fail goto*//*Label 1169*/ 28944, // Rule ID 16203 //
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_s64,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::GR64RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::GR64RegClassID,
      // (cttz_zero_undef:{ *:[i64] } GR64:{ *:[i64] }:$src)  =>  (BSF64rr:{ *:[i64] }:{ *:[i32] } GR64:{ *:[i64] }:$src)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::BSF64rr,
      GIR_AddImplicitDef, /*InsnID*/0, X86::EFLAGS,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 16203,
      GIR_Done,
    // Label 1169: @28944
    GIM_Reject,
    // Label 1166: @28945
    GIM_Reject,
    // Label 31: @28946
    GIM_SwitchType, /*MI*/0, /*Op*/0, /*[*/8, 20, /*)*//*default:*//*Label 1176*/ 29108,
    /*GILLT_v2s64*//*Label 1170*/ 28964, 0,
    /*GILLT_v4s32*//*Label 1171*/ 28988,
    /*GILLT_v4s64*//*Label 1172*/ 29012, 0, 0,
    /*GILLT_v8s32*//*Label 1173*/ 29036,
    /*GILLT_v8s64*//*Label 1174*/ 29060, 0, 0, 0,
    /*GILLT_v16s32*//*Label 1175*/ 29084,
    // Label 1170: @28964
    GIM_Try, /*On fail goto*//*Label 1177*/ 28987, // Rule ID 10456 //
      GIM_CheckFeatures, GIFBS_HasCDI_HasVLX,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v2s64,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128XRegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR128XRegClassID,
      // (ctlz:{ *:[v2i64] } VR128X:{ *:[v2i64] }:$src1)  =>  (VPLZCNTQZ128rr:{ *:[v2i64] } VR128X:{ *:[v2i64] }:$src1)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VPLZCNTQZ128rr,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 10456,
      GIR_Done,
    // Label 1177: @28987
    GIM_Reject,
    // Label 1171: @28988
    GIM_Try, /*On fail goto*//*Label 1178*/ 29011, // Rule ID 10483 //
      GIM_CheckFeatures, GIFBS_HasCDI_HasVLX,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v4s32,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128XRegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR128XRegClassID,
      // (ctlz:{ *:[v4i32] } VR128X:{ *:[v4i32] }:$src1)  =>  (VPLZCNTDZ128rr:{ *:[v4i32] } VR128X:{ *:[v4i32] }:$src1)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VPLZCNTDZ128rr,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 10483,
      GIR_Done,
    // Label 1178: @29011
    GIM_Reject,
    // Label 1172: @29012
    GIM_Try, /*On fail goto*//*Label 1179*/ 29035, // Rule ID 10447 //
      GIM_CheckFeatures, GIFBS_HasCDI_HasVLX,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v4s64,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR256XRegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR256XRegClassID,
      // (ctlz:{ *:[v4i64] } VR256X:{ *:[v4i64] }:$src1)  =>  (VPLZCNTQZ256rr:{ *:[v4i64] } VR256X:{ *:[v4i64] }:$src1)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VPLZCNTQZ256rr,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 10447,
      GIR_Done,
    // Label 1179: @29035
    GIM_Reject,
    // Label 1173: @29036
    GIM_Try, /*On fail goto*//*Label 1180*/ 29059, // Rule ID 10474 //
      GIM_CheckFeatures, GIFBS_HasCDI_HasVLX,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v8s32,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR256XRegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR256XRegClassID,
      // (ctlz:{ *:[v8i32] } VR256X:{ *:[v8i32] }:$src1)  =>  (VPLZCNTDZ256rr:{ *:[v8i32] } VR256X:{ *:[v8i32] }:$src1)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VPLZCNTDZ256rr,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 10474,
      GIR_Done,
    // Label 1180: @29059
    GIM_Reject,
    // Label 1174: @29060
    GIM_Try, /*On fail goto*//*Label 1181*/ 29083, // Rule ID 10438 //
      GIM_CheckFeatures, GIFBS_HasCDI,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v8s64,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR512RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR512RegClassID,
      // (ctlz:{ *:[v8i64] } VR512:{ *:[v8i64] }:$src1)  =>  (VPLZCNTQZrr:{ *:[v8i64] } VR512:{ *:[v8i64] }:$src1)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VPLZCNTQZrr,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 10438,
      GIR_Done,
    // Label 1181: @29083
    GIM_Reject,
    // Label 1175: @29084
    GIM_Try, /*On fail goto*//*Label 1182*/ 29107, // Rule ID 10465 //
      GIM_CheckFeatures, GIFBS_HasCDI,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v16s32,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR512RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR512RegClassID,
      // (ctlz:{ *:[v16i32] } VR512:{ *:[v16i32] }:$src1)  =>  (VPLZCNTDZrr:{ *:[v16i32] } VR512:{ *:[v16i32] }:$src1)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VPLZCNTDZrr,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 10465,
      GIR_Done,
    // Label 1182: @29107
    GIM_Reject,
    // Label 1176: @29108
    GIM_Reject,
    // Label 32: @29109
    GIM_SwitchType, /*MI*/0, /*Op*/0, /*[*/8, 25, /*)*//*default:*//*Label 1195*/ 29420,
    /*GILLT_v2s64*//*Label 1183*/ 29132, 0,
    /*GILLT_v4s32*//*Label 1184*/ 29156,
    /*GILLT_v4s64*//*Label 1185*/ 29180, 0,
    /*GILLT_v8s16*//*Label 1186*/ 29204,
    /*GILLT_v8s32*//*Label 1187*/ 29228,
    /*GILLT_v8s64*//*Label 1188*/ 29252, 0,
    /*GILLT_v16s8*//*Label 1189*/ 29276,
    /*GILLT_v16s16*//*Label 1190*/ 29300,
    /*GILLT_v16s32*//*Label 1191*/ 29324, 0,
    /*GILLT_v32s8*//*Label 1192*/ 29348,
    /*GILLT_v32s16*//*Label 1193*/ 29372, 0,
    /*GILLT_v64s8*//*Label 1194*/ 29396,
    // Label 1183: @29132
    GIM_Try, /*On fail goto*//*Label 1196*/ 29155, // Rule ID 10564 //
      GIM_CheckFeatures, GIFBS_HasVLX_HasVPOPCNTDQ,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v2s64,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128XRegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR128XRegClassID,
      // (ctpop:{ *:[v2i64] } VR128X:{ *:[v2i64] }:$src1)  =>  (VPOPCNTQZ128rr:{ *:[v2i64] } VR128X:{ *:[v2i64] }:$src1)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VPOPCNTQZ128rr,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 10564,
      GIR_Done,
    // Label 1196: @29155
    GIM_Reject,
    // Label 1184: @29156
    GIM_Try, /*On fail goto*//*Label 1197*/ 29179, // Rule ID 10591 //
      GIM_CheckFeatures, GIFBS_HasVLX_HasVPOPCNTDQ,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v4s32,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128XRegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR128XRegClassID,
      // (ctpop:{ *:[v4i32] } VR128X:{ *:[v4i32] }:$src1)  =>  (VPOPCNTDZ128rr:{ *:[v4i32] } VR128X:{ *:[v4i32] }:$src1)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VPOPCNTDZ128rr,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 10591,
      GIR_Done,
    // Label 1197: @29179
    GIM_Reject,
    // Label 1185: @29180
    GIM_Try, /*On fail goto*//*Label 1198*/ 29203, // Rule ID 10555 //
      GIM_CheckFeatures, GIFBS_HasVLX_HasVPOPCNTDQ,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v4s64,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR256XRegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR256XRegClassID,
      // (ctpop:{ *:[v4i64] } VR256X:{ *:[v4i64] }:$src1)  =>  (VPOPCNTQZ256rr:{ *:[v4i64] } VR256X:{ *:[v4i64] }:$src1)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VPOPCNTQZ256rr,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 10555,
      GIR_Done,
    // Label 1198: @29203
    GIM_Reject,
    // Label 1186: @29204
    GIM_Try, /*On fail goto*//*Label 1199*/ 29227, // Rule ID 11672 //
      GIM_CheckFeatures, GIFBS_HasBITALG_HasVLX,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v8s16,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128XRegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR128XRegClassID,
      // (ctpop:{ *:[v8i16] } VR128X:{ *:[v8i16] }:$src1)  =>  (VPOPCNTWZ128rr:{ *:[v8i16] } VR128X:{ *:[v8i16] }:$src1)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VPOPCNTWZ128rr,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 11672,
      GIR_Done,
    // Label 1199: @29227
    GIM_Reject,
    // Label 1187: @29228
    GIM_Try, /*On fail goto*//*Label 1200*/ 29251, // Rule ID 10582 //
      GIM_CheckFeatures, GIFBS_HasVLX_HasVPOPCNTDQ,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v8s32,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR256XRegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR256XRegClassID,
      // (ctpop:{ *:[v8i32] } VR256X:{ *:[v8i32] }:$src1)  =>  (VPOPCNTDZ256rr:{ *:[v8i32] } VR256X:{ *:[v8i32] }:$src1)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VPOPCNTDZ256rr,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 10582,
      GIR_Done,
    // Label 1200: @29251
    GIM_Reject,
    // Label 1188: @29252
    GIM_Try, /*On fail goto*//*Label 1201*/ 29275, // Rule ID 10546 //
      GIM_CheckFeatures, GIFBS_HasVPOPCNTDQ,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v8s64,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR512RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR512RegClassID,
      // (ctpop:{ *:[v8i64] } VR512:{ *:[v8i64] }:$src1)  =>  (VPOPCNTQZrr:{ *:[v8i64] } VR512:{ *:[v8i64] }:$src1)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VPOPCNTQZrr,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 10546,
      GIR_Done,
    // Label 1201: @29275
    GIM_Reject,
    // Label 1189: @29276
    GIM_Try, /*On fail goto*//*Label 1202*/ 29299, // Rule ID 11654 //
      GIM_CheckFeatures, GIFBS_HasBITALG_HasVLX,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v16s8,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128XRegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR128XRegClassID,
      // (ctpop:{ *:[v16i8] } VR128X:{ *:[v16i8] }:$src1)  =>  (VPOPCNTBZ128rr:{ *:[v16i8] } VR128X:{ *:[v16i8] }:$src1)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VPOPCNTBZ128rr,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 11654,
      GIR_Done,
    // Label 1202: @29299
    GIM_Reject,
    // Label 1190: @29300
    GIM_Try, /*On fail goto*//*Label 1203*/ 29323, // Rule ID 11666 //
      GIM_CheckFeatures, GIFBS_HasBITALG_HasVLX,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v16s16,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR256XRegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR256XRegClassID,
      // (ctpop:{ *:[v16i16] } VR256X:{ *:[v16i16] }:$src1)  =>  (VPOPCNTWZ256rr:{ *:[v16i16] } VR256X:{ *:[v16i16] }:$src1)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VPOPCNTWZ256rr,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 11666,
      GIR_Done,
    // Label 1203: @29323
    GIM_Reject,
    // Label 1191: @29324
    GIM_Try, /*On fail goto*//*Label 1204*/ 29347, // Rule ID 10573 //
      GIM_CheckFeatures, GIFBS_HasVPOPCNTDQ,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v16s32,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR512RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR512RegClassID,
      // (ctpop:{ *:[v16i32] } VR512:{ *:[v16i32] }:$src1)  =>  (VPOPCNTDZrr:{ *:[v16i32] } VR512:{ *:[v16i32] }:$src1)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VPOPCNTDZrr,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 10573,
      GIR_Done,
    // Label 1204: @29347
    GIM_Reject,
    // Label 1192: @29348
    GIM_Try, /*On fail goto*//*Label 1205*/ 29371, // Rule ID 11648 //
      GIM_CheckFeatures, GIFBS_HasBITALG_HasVLX,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v32s8,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR256XRegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR256XRegClassID,
      // (ctpop:{ *:[v32i8] } VR256X:{ *:[v32i8] }:$src1)  =>  (VPOPCNTBZ256rr:{ *:[v32i8] } VR256X:{ *:[v32i8] }:$src1)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VPOPCNTBZ256rr,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 11648,
      GIR_Done,
    // Label 1205: @29371
    GIM_Reject,
    // Label 1193: @29372
    GIM_Try, /*On fail goto*//*Label 1206*/ 29395, // Rule ID 11660 //
      GIM_CheckFeatures, GIFBS_HasBITALG,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v32s16,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR512RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR512RegClassID,
      // (ctpop:{ *:[v32i16] } VR512:{ *:[v32i16] }:$src1)  =>  (VPOPCNTWZrr:{ *:[v32i16] } VR512:{ *:[v32i16] }:$src1)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VPOPCNTWZrr,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 11660,
      GIR_Done,
    // Label 1206: @29395
    GIM_Reject,
    // Label 1194: @29396
    GIM_Try, /*On fail goto*//*Label 1207*/ 29419, // Rule ID 11642 //
      GIM_CheckFeatures, GIFBS_HasBITALG,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v64s8,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR512RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR512RegClassID,
      // (ctpop:{ *:[v64i8] } VR512:{ *:[v64i8] }:$src1)  =>  (VPOPCNTBZrr:{ *:[v64i8] } VR512:{ *:[v64i8] }:$src1)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VPOPCNTBZrr,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 11642,
      GIR_Done,
    // Label 1207: @29419
    GIM_Reject,
    // Label 1195: @29420
    GIM_Reject,
    // Label 33: @29421
    GIM_SwitchType, /*MI*/0, /*Op*/0, /*[*/2, 5, /*)*//*default:*//*Label 1211*/ 29510,
    /*GILLT_s16*//*Label 1208*/ 29430,
    /*GILLT_s32*//*Label 1209*/ 29466,
    /*GILLT_s64*//*Label 1210*/ 29488,
    // Label 1208: @29430
    GIM_Try, /*On fail goto*//*Label 1212*/ 29465, // Rule ID 16207 //
      GIM_CheckFeatures, GIFBS_HasMOVBE,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_s16,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::GR16RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::GR16RegClassID,
      // (bswap:{ *:[i16] } GR16:{ *:[i16] }:$src)  =>  (ROL16ri:{ *:[i16] }:{ *:[i32] } GR16:{ *:[i16] }:$src, 8:{ *:[i8] })
      GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::ROL16ri,
      GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
      GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // src
      GIR_AddImm, /*InsnID*/0, /*Imm*/8,
      GIR_EraseFromParent, /*InsnID*/0,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 16207,
      GIR_Done,
    // Label 1212: @29465
    GIM_Reject,
    // Label 1209: @29466
    GIM_Try, /*On fail goto*//*Label 1213*/ 29487, // Rule ID 5 //
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_s32,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::GR32RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::GR32RegClassID,
      // (bswap:{ *:[i32] } GR32:{ *:[i32] }:$src)  =>  (BSWAP32r:{ *:[i32] } GR32:{ *:[i32] }:$src)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::BSWAP32r,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 5,
      GIR_Done,
    // Label 1213: @29487
    GIM_Reject,
    // Label 1210: @29488
    GIM_Try, /*On fail goto*//*Label 1214*/ 29509, // Rule ID 6 //
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_s64,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::GR64RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::GR64RegClassID,
      // (bswap:{ *:[i64] } GR64:{ *:[i64] }:$src)  =>  (BSWAP64r:{ *:[i64] } GR64:{ *:[i64] }:$src)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::BSWAP64r,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 6,
      GIR_Done,
    // Label 1214: @29509
    GIM_Reject,
    // Label 1211: @29510
    GIM_Reject,
    // Label 34: @29511
    GIM_Reject,
    };
  return MatchTable0;
}
#endif // ifdef GET_GLOBALISEL_IMPL
#ifdef GET_GLOBALISEL_PREDICATES_DECL
PredicateBitset AvailableModuleFeatures;
mutable PredicateBitset AvailableFunctionFeatures;
PredicateBitset getAvailableFeatures() const {
  return AvailableModuleFeatures | AvailableFunctionFeatures;
}
PredicateBitset
computeAvailableModuleFeatures(const X86Subtarget *Subtarget) const;
PredicateBitset
computeAvailableFunctionFeatures(const X86Subtarget *Subtarget,
                                 const MachineFunction *MF) const;
#endif // ifdef GET_GLOBALISEL_PREDICATES_DECL
#ifdef GET_GLOBALISEL_PREDICATES_INIT
AvailableModuleFeatures(computeAvailableModuleFeatures(&STI)),
AvailableFunctionFeatures()
#endif // ifdef GET_GLOBALISEL_PREDICATES_INIT
