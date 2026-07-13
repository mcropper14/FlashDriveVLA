using Cxx = import "./include/c++.capnp";
$Cxx.namespace("cereal");

@0xb526ba661d550a59;

# custom.capnp: a home for empty structs reserved for custom forks
# These structs are guaranteed to remain reserved and empty in mainline
# cereal, so use these if you want custom events in your fork.

# DO rename the structs
# DON'T change the identifier (e.g. @0x81c2f05a394cf4af)

struct ReasonedTrajectoryPlan @0x81c2f05a394cf4af {
  frameId @0 :UInt32;
  modelMonoTime @1 :UInt64;
  planValid @2 :Bool;

  scene @3 :Text;
  evidence @4 :Text;
  meta @5 :Text;
  branch @6 :Text;
  latBiasM @7 :Float32;
  speedCapMps @8 :Float32;
  stopS @9 :Float32;
  avoid @10 :Text;
  weights @11 :Text;
  confidence @12 :Float32;

  generatedTokenCount @13 :UInt16;
  cameraToSceneBoardMs @14 :Float32;
  sceneBoardToVlmPrefillMs @15 :Float32;
  vlmDecodeMs @16 :Float32;
  rtpParseMs @17 :Float32;
  pathSynthMs @18 :Float32;
  publishAgeMs @19 :Float32;
  controlConsumedAgeMs @20 :Float32;
  deadlineMissCount @21 :UInt32;
  invalidRtpCount @22 :UInt32;

  vlmChangedPathMeters @23 :Float32;
  vlmChangedSpeedMps @24 :Float32;
  selectedCandidate @25 :Text;
  desiredCurvature @26 :Float32;
  vlmBackend @27 :Text;
  rtpText @28 :Text;
  invalidReason @29 :Text;
  rtpSourceFrameId @30 :Int32;
  rtpAgeFrames @31 :Int16;
}

struct CustomReserved1 @0xaedffd8f31e7b55d {
}

struct CustomReserved2 @0xf35cc4560bbf6ec2 {
}

struct CustomReserved3 @0xda96579883444c35 {
}

struct CustomReserved4 @0x80ae746ee2596b11 {
}

struct CustomReserved5 @0xa5cd762cd951a455 {
}

struct CustomReserved6 @0xf98d843bfd7004a3 {
}

struct CustomReserved7 @0xb86e6369214c01c8 {
}

struct CustomReserved8 @0xf416ec09499d9d19 {
}

struct CustomReserved9 @0xa1680744031fdb2d {
}

struct CustomReserved10 @0xcb9fd56c7057593a {
}

struct CustomReserved11 @0xc2243c65e0340384 {
}

struct CustomReserved12 @0x9ccdc8676701b412 {
}

struct CustomReserved13 @0xcd96dafb67a082d0 {
}

struct CustomReserved14 @0xb057204d7deadf3f {
}

struct CustomReserved15 @0xbd443b539493bc68 {
}

struct CustomReserved16 @0xfc6241ed8877b611 {
}

struct CustomReserved17 @0xa30662f84033036c {
}

struct CustomReserved18 @0xc86a3d38d13eb3ef {
}

struct CustomReserved19 @0xa4f1eb3323f5f582 {
}
