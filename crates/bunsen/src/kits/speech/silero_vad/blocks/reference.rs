//! Reference ONNX Exported Silero VAD Model.

use burn::{
    Tensor,
    module::{Module, Param, ParamId},
    nn::{
        Linear, LinearConfig, LinearLayout, PaddingConfig1d,
        conv::{Conv1d, Conv1dConfig},
    },
    prelude::{Backend, Int, s},
    tensor::{
        Bytes,
        activation::{relu, sigmoid, tanh},
        ops::PadMode,
    },
};
use burn_store::{BurnpackStore, ModuleSnapshot};

/// Reference model for Silero VAD.
#[derive(Module, Debug)]
pub struct ReferenceVAD<B: Backend> {
    constant32: Param<Tensor<B, 1, Int>>,
    constant41: Param<Tensor<B, 1, Int>>,
    constant42: Param<Tensor<B, 1>>,
    conv1d37: Conv1d<B>,
    conv1d38: Conv1d<B>,
    conv1d39: Conv1d<B>,
    conv1d40: Conv1d<B>,
    conv1d41: Conv1d<B>,
    linear13: Linear<B>,
    linear14: Linear<B>,
    conv1d42: Conv1d<B>,
    conv1d43: Conv1d<B>,
    conv1d44: Conv1d<B>,
    conv1d45: Conv1d<B>,
    conv1d46: Conv1d<B>,
    conv1d47: Conv1d<B>,
    linear15: Linear<B>,
    linear16: Linear<B>,
    conv1d48: Conv1d<B>,
}

impl<B: Backend> ReferenceVAD<B> {
    /// Load model weights from a burnpack file.
    pub fn from_file(
        file: &str,
        device: &B::Device,
    ) -> Self {
        let mut model = Self::new(device);
        let mut store = BurnpackStore::from_file(file);
        model
            .load_from(&mut store)
            .expect("Failed to load burnpack file");
        model
    }

    /// Load model weights from in-memory bytes.
    ///
    /// The bytes must be the contents of a `.bpk` file.
    pub fn from_bytes(
        bytes: Bytes,
        device: &B::Device,
    ) -> Self {
        let mut model = Self::new(device);
        let mut store = BurnpackStore::from_bytes(Some(bytes));
        model
            .load_from(&mut store)
            .expect("Failed to load burnpack bytes");
        model
    }
}

impl<B: Backend> ReferenceVAD<B> {
    /// Build a new reference model.
    #[allow(unused_variables)]
    pub fn new(device: &B::Device) -> Self {
        let constant32: Param<Tensor<B, 1, Int>> = Param::uninitialized(
            ParamId::new(),
            move |device, _require_grad| Tensor::<B, 1, Int>::from_data([0i64], device),
            device.clone(),
            false,
            [1].into(),
        );
        let constant41: Param<Tensor<B, 1, Int>> = Param::uninitialized(
            ParamId::new(),
            move |device, _require_grad| Tensor::<B, 1, Int>::from_data([1i64], device),
            device.clone(),
            false,
            [1].into(),
        );
        let constant42: Param<Tensor<B, 1>> = Param::uninitialized(
            ParamId::new(),
            move |device, _require_grad| Tensor::<B, 1>::from_data([2f64], device),
            device.clone(),
            false,
            [1].into(),
        );
        let conv1d37 = Conv1dConfig::new(1, 258, 256)
            .with_stride(128)
            .with_padding(PaddingConfig1d::Valid)
            .with_dilation(1)
            .with_groups(1)
            .with_bias(false)
            .init(device);
        let conv1d38 = Conv1dConfig::new(129, 128, 3)
            .with_stride(1)
            .with_padding(PaddingConfig1d::Explicit(1, 1))
            .with_dilation(1)
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv1d39 = Conv1dConfig::new(128, 64, 3)
            .with_stride(2)
            .with_padding(PaddingConfig1d::Explicit(1, 1))
            .with_dilation(1)
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv1d40 = Conv1dConfig::new(64, 64, 3)
            .with_stride(2)
            .with_padding(PaddingConfig1d::Explicit(1, 1))
            .with_dilation(1)
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv1d41 = Conv1dConfig::new(64, 128, 3)
            .with_stride(1)
            .with_padding(PaddingConfig1d::Explicit(1, 1))
            .with_dilation(1)
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let linear13 = LinearConfig::new(128, 512)
            .with_bias(true)
            .with_layout(LinearLayout::Col)
            .init(device);
        let linear14 = LinearConfig::new(128, 512)
            .with_bias(true)
            .with_layout(LinearLayout::Col)
            .init(device);
        let conv1d42 = Conv1dConfig::new(128, 1, 1)
            .with_stride(1)
            .with_padding(PaddingConfig1d::Valid)
            .with_dilation(1)
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv1d43 = Conv1dConfig::new(1, 130, 128)
            .with_stride(64)
            .with_padding(PaddingConfig1d::Valid)
            .with_dilation(1)
            .with_groups(1)
            .with_bias(false)
            .init(device);
        let conv1d44 = Conv1dConfig::new(65, 128, 3)
            .with_stride(1)
            .with_padding(PaddingConfig1d::Explicit(1, 1))
            .with_dilation(1)
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv1d45 = Conv1dConfig::new(128, 64, 3)
            .with_stride(2)
            .with_padding(PaddingConfig1d::Explicit(1, 1))
            .with_dilation(1)
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv1d46 = Conv1dConfig::new(64, 64, 3)
            .with_stride(2)
            .with_padding(PaddingConfig1d::Explicit(1, 1))
            .with_dilation(1)
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv1d47 = Conv1dConfig::new(64, 128, 3)
            .with_stride(1)
            .with_padding(PaddingConfig1d::Explicit(1, 1))
            .with_dilation(1)
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let linear15 = LinearConfig::new(128, 512)
            .with_bias(true)
            .with_layout(LinearLayout::Col)
            .init(device);
        let linear16 = LinearConfig::new(128, 512)
            .with_bias(true)
            .with_layout(LinearLayout::Col)
            .init(device);
        let conv1d48 = Conv1dConfig::new(128, 1, 1)
            .with_stride(1)
            .with_padding(PaddingConfig1d::Valid)
            .with_dilation(1)
            .with_groups(1)
            .with_bias(true)
            .init(device);
        Self {
            constant32,
            constant41,
            constant42,
            conv1d37,
            conv1d38,
            conv1d39,
            conv1d40,
            conv1d41,
            linear13,
            linear14,
            conv1d42,
            conv1d43,
            conv1d44,
            conv1d45,
            conv1d46,
            conv1d47,
            linear15,
            linear16,
            conv1d48,
        }
    }

    /// Run the module.
    #[allow(clippy::let_and_return, clippy::approx_constant)]
    pub fn forward(
        &self,
        input: Tensor<B, 2>,
        sr: usize,
        state: Tensor<B, 3>,
    ) -> (Tensor<B, 2>, Tensor<B, 3>) {
        match sr {
            16000 => self.forward_16khz(input, state),
            8000 => self.forward_8khz(input, state),
            _ => panic!("unsupported sample rate: {sr}"),
        }
    }

    /// (cell, hidden)
    fn unpack_state(state: Tensor<B, 3>) -> (Tensor<B, 2>, Tensor<B, 2>) {
        let hidden = state.clone().slice_dim(0, 0).squeeze_dim::<2usize>(0);
        let cell = state.slice_dim(0, 1).squeeze_dim::<2usize>(0);
        (cell, hidden)
    }

    /// Stacks `(cell, hidden)` into a packed `[2, batch, hidden]` state.
    fn pack_state(
        cell: Tensor<B, 2>,
        hidden: Tensor<B, 2>,
    ) -> Tensor<B, 3> {
        Tensor::stack(vec![hidden, cell], 0)
    }

    /// Frame Features, 16khz
    pub fn frame_features_16khz(
        &self,
        input: Tensor<B, 2>,
    ) -> Tensor<B, 2> {
        let x = input.pad([(0, 0), (0, 64)], PadMode::Reflect);
        let x: Tensor<B, 3> = x.unsqueeze_dim::<3>(1);

        let [real_2, imag_2] = self
            .conv1d37
            .forward(x)
            .square()
            .chunk(2, 1)
            .try_into()
            .unwrap();
        let x = (real_2 + imag_2).sqrt();

        // Encoder
        let x = self.conv1d38.forward(x);
        let x = relu(x);
        let x = self.conv1d39.forward(x);
        let x = relu(x);
        let x = self.conv1d40.forward(x);
        let x = relu(x);
        let x = self.conv1d41.forward(x);
        let x = relu(x);

        x.slice_dim(2, 0).squeeze_dim::<2usize>(2)
    }

    /// Run the module.
    #[allow(clippy::let_and_return, clippy::approx_constant)]
    pub fn forward_16khz(
        &self,
        input: Tensor<B, 2>,
        state: Tensor<B, 3>,
    ) -> (Tensor<B, 2>, Tensor<B, 3>) {
        let input = input.clone();
        let state = state.clone();

        let features = self.frame_features_16khz(input);

        let (cell, hidden) = Self::unpack_state(state);

        let gates = self.linear13.forward(hidden) + self.linear14.forward(features);

        let [g_i, g_f, g_c, g_o] = gates.chunk(4, 1).try_into().unwrap();

        let input_values = sigmoid(g_i);
        let forget_values = sigmoid(g_f);
        let candidate_cell_values = tanh(g_c);
        let output_values = sigmoid(g_o);

        let cell = (forget_values * cell) + (input_values * candidate_cell_values);
        let hidden = output_values * tanh(cell.clone());

        let state = Self::pack_state(cell, hidden.clone());

        // output head
        let x: Tensor<B, 3> = hidden.unsqueeze_dim::<3>(2);
        let x = relu(x);
        let x = self.conv1d42.forward(x);
        let x = sigmoid(x);
        let x = x.squeeze_dims::<2>(&[1]);
        let x = x.mean_dim(1);
        // let x: Tensor<B, 2> = x.squeeze_dims::<1>(&[1]).unsqueeze_dims::<2>(&[1]);

        (x, state)
    }

    /// Run the module.
    #[allow(clippy::let_and_return, clippy::approx_constant)]
    pub fn forward_8khz(
        &self,
        input: Tensor<B, 2>,
        state: Tensor<B, 3>,
    ) -> (Tensor<B, 2>, Tensor<B, 3>) {
        let input = input.clone();
        let state = state.clone();

        let pad8_out1 = input.pad([(0usize, 0usize), (0usize, 32usize)], PadMode::Reflect);
        let unsqueeze36_out1: Tensor<B, 3> = pad8_out1.unsqueeze_dims::<3>(&[1]);
        let conv1d43_out1 = self.conv1d43.forward(unsqueeze36_out1);
        let slice15_out1 = conv1d43_out1.clone().slice(s![.., 0..65, ..]);
        let slice16_out1 = conv1d43_out1.slice(s![.., 65.., ..]);
        let pow15_out1 = slice15_out1.square();
        let pow16_out1 = slice16_out1.square();
        let sqrt8_out1 = (pow15_out1 + pow16_out1).sqrt();
        let conv1d44_out1 = self.conv1d44.forward(sqrt8_out1);
        let relu36_out1 = relu(conv1d44_out1);
        let conv1d45_out1 = self.conv1d45.forward(relu36_out1);
        let relu37_out1 = relu(conv1d45_out1);
        let conv1d46_out1 = self.conv1d46.forward(relu37_out1);
        let relu38_out1 = relu(conv1d46_out1);
        let conv1d47_out1 = self.conv1d47.forward(relu38_out1);
        let relu39_out1 = relu(conv1d47_out1);
        let features = {
            let sliced = relu39_out1.slice(s![.., .., 0i64]);
            sliced.squeeze_dim::<2usize>(2)
        };

        let gather25_out1 = {
            let sliced = state.clone().slice(s![0i64, .., ..]);
            sliced.squeeze_dim::<2usize>(0)
        };
        let gather26_out1 = {
            let sliced = state.slice(s![1i64, .., ..]);
            sliced.squeeze_dim::<2usize>(0)
        };
        let linear15_out1 = self.linear15.forward(gather25_out1);
        let linear16_out1 = self.linear16.forward(features);
        let add23_out1 = linear15_out1.add(linear16_out1);
        let split_tensors = add23_out1.split_with_sizes([128, 128, 128, 128].into(), 1);
        let [split8_out1, split8_out2, split8_out3, split8_out4] =
            split_tensors.try_into().unwrap();
        let sigmoid29_out1 = sigmoid(split8_out1);
        let sigmoid30_out1 = sigmoid(split8_out2);
        let tanh15_out1 = split8_out3.tanh();
        let sigmoid31_out1 = sigmoid(split8_out4);
        let mul22_out1 = sigmoid30_out1.mul(gather26_out1);
        let mul23_out1 = sigmoid29_out1.mul(tanh15_out1);
        let add24_out1 = mul22_out1.add(mul23_out1);
        let tanh16_out1 = add24_out1.clone().tanh();
        let mul24_out1 = sigmoid31_out1.mul(tanh16_out1);
        let unsqueeze37_out1: Tensor<B, 3> = mul24_out1.clone().unsqueeze_dims::<3>(&[-1]);
        let unsqueeze38_out1: Tensor<B, 3> = mul24_out1.unsqueeze_dims::<3>(&[0]);
        let unsqueeze39_out1: Tensor<B, 3> = add24_out1.unsqueeze_dims::<3>(&[0]);
        let concat8_out1 = Tensor::cat([unsqueeze38_out1, unsqueeze39_out1].into(), 0);
        let relu40_out1 = relu(unsqueeze37_out1);
        let conv1d48_out1 = self.conv1d48.forward(relu40_out1);
        let sigmoid32_out1 = sigmoid(conv1d48_out1);
        let squeeze8_out1 = sigmoid32_out1.squeeze_dims::<2>(&[1]);
        let reducemean8_out1 = { squeeze8_out1.mean_dim(1usize).squeeze_dims::<1usize>(&[1]) };
        let unsqueeze40_out1: Tensor<B, 2> = reducemean8_out1.unsqueeze_dims::<2>(&[1]);
        (unsqueeze40_out1, concat8_out1)
    }
}

#[cfg(test)]
mod tests {
    use std::{fs, path::PathBuf};

    use burn::{
        Tensor,
        tensor::{Distribution, TensorData, Tolerance, backend::BackendTypes},
    };

    use super::*;
    use crate::{
        errors::*, kits::speech::silero_vad::blocks::SileroVad16x8,
        support::testing::PerformanceBackend,
    };

    /// A valid chunk length for the given sample rate (standard Silero chunk).
    pub fn chunk_samples(sample_rate: usize) -> usize {
        match sample_rate {
            16000 => 512,
            8000 => 256,
            other => panic!("no test chunk for {other}"),
        }
    }

    fn silero_burnpack_path() -> PathBuf {
        PathBuf::from("silero_vad_op18_ifless.bpk")
    }

    #[test]
    fn test_load_forward_pretrained() {
        type B = PerformanceBackend;
        type F = <B as BackendTypes>::FloatElem;

        let path = silero_burnpack_path();
        let device = Default::default();

        let s_mod: SileroVad16x8<B> =
            SileroVad16x8::load_from_burnpack(&path, &device).ok_or_panic();

        let r_mod: ReferenceVAD<B> = ReferenceVAD::from_file(path.to_str().unwrap(), &device);

        let batch = 2;
        let state = Tensor::zeros([2, batch, 128], &device);

        // 16khz
        {
            let sample_rate = 16000;
            let input = Tensor::<B, 2>::random(
                [batch, chunk_samples(sample_rate)],
                Distribution::Default,
                &device,
            );

            let (s_out, s_state) = s_mod.forward(input.clone(), sample_rate, state.clone());
            let (r_out, r_state) = r_mod.forward(input.clone(), sample_rate, state.clone());

            s_out
                .to_data()
                .assert_approx_eq::<F>(&r_out.to_data(), Tolerance::default());

            s_state
                .to_data()
                .assert_approx_eq::<F>(&r_state.to_data(), Tolerance::default());
        }

        // 8khz
        {
            let sample_rate = 8000;

            let input = Tensor::<B, 2>::random(
                [batch, chunk_samples(sample_rate)],
                Distribution::Default,
                &device,
            );

            let (s_out, s_state) = s_mod.forward(input.clone(), sample_rate, state.clone());
            let (r_out, r_state) = r_mod.forward(input.clone(), sample_rate, state.clone());

            s_out
                .to_data()
                .assert_approx_eq::<F>(&r_out.to_data(), Tolerance::default());

            s_state
                .to_data()
                .assert_approx_eq::<F>(&r_state.to_data(), Tolerance::default());
        }
    }

    const REF_DATA: &[f32] = &[
        0.008708502,
        0.0066900332,
        0.0043916213,
        0.0066658775,
        0.008642789,
        0.010985533,
        0.010536938,
        0.0075576557,
        0.0062581305,
        0.0061512254,
        0.00477772,
        0.004033637,
        0.0070916056,
        0.03450967,
        0.013106528,
        0.046747923,
        0.080566004,
        0.23036888,
        0.97280335,
        0.9964134,
        0.99900925,
        0.999166,
        0.9997209,
        0.9993268,
        0.9996259,
        0.99992454,
        0.99994874,
        0.99982893,
        0.99977726,
        0.9997578,
        0.9989697,
        0.9989348,
        0.9976587,
        0.9351481,
        0.38673526,
        0.1300675,
        0.13164346,
        0.98511654,
        0.99893516,
        0.9996038,
        0.9997385,
        0.9996371,
        0.99988663,
        0.9997801,
        0.99966395,
        0.9996339,
        0.9996271,
        0.9994702,
        0.99828607,
        0.9922287,
        0.87959325,
        0.6737317,
        0.9918761,
        0.9997589,
        0.99995625,
        0.9999641,
        0.9999281,
        0.9999119,
        0.99983084,
        0.9995154,
        0.99992526,
        0.99993086,
        0.9999552,
        0.9999459,
        0.9999758,
        0.9999715,
        0.9999294,
        0.9998354,
        0.99939144,
        0.97805417,
        0.47254163,
        0.10105009,
        0.026438193,
        0.01319484,
        0.009913754,
        0.0076286644,
        0.0065150913,
        0.7402992,
        0.6183067,
        0.9958301,
        0.9999403,
        0.99996495,
        0.9999733,
        0.9999727,
        0.9999683,
        0.9999751,
        0.9999442,
        0.9999186,
        0.99975747,
        0.9995245,
        0.99959236,
        0.99914634,
        0.9981679,
        0.9973552,
        0.9996904,
        0.99997926,
        0.9999845,
        0.99998796,
        0.99997175,
        0.99998367,
        0.9999751,
        0.9999695,
        0.99996316,
        0.999938,
        0.9998869,
        0.99929655,
        0.9805439,
        0.86452705,
        0.99956554,
        0.9999819,
        0.9999776,
        0.9999337,
        0.9998802,
        0.99959224,
        0.99272597,
        0.9997427,
        0.99924743,
        0.99935704,
        0.9998995,
        0.9999819,
        0.9997589,
        0.9984334,
        0.9987544,
        0.9943944,
        0.952861,
        0.56911045,
        0.31985438,
        0.43110812,
        0.5612931,
        0.5111101,
        0.30298734,
        0.08823484,
        0.06037989,
        0.037814807,
        0.03302366,
        0.028889744,
        0.60302675,
        0.99395835,
        0.99995685,
        0.99992716,
        0.9999573,
        0.9999664,
        0.9999653,
        0.9999908,
        0.99999285,
        0.9999794,
        0.9998393,
        0.99900264,
        0.99821067,
        0.9938863,
        0.9997813,
        0.9999708,
        0.9999237,
        0.99995697,
        0.9999858,
        0.9999658,
        0.9999658,
        0.99985766,
        0.9991116,
        0.9970209,
        0.99971753,
        0.99997973,
        0.99998057,
        0.99997616,
        0.99994993,
        0.99984384,
        0.99983203,
        0.9999269,
        0.99999034,
        0.99998987,
        0.9998902,
        0.9999757,
        0.99996865,
        0.9999068,
        0.999746,
        0.9976006,
        0.99497837,
        0.83459026,
        0.3942421,
        0.10653588,
        0.042795543,
        0.026254144,
        0.01763668,
        0.013192855,
        0.14191277,
        0.9169647,
        0.99920577,
        0.99996495,
        0.99996865,
        0.9999367,
        0.99991846,
        0.99995434,
        0.99994254,
        0.99995136,
        0.99992025,
        0.9998895,
        0.9999658,
        0.9997867,
        0.9996668,
        0.99902594,
        0.98712105,
        0.58031785,
        0.15301901,
        0.095074736,
        0.047488555,
        0.93318677,
        0.99890256,
        0.99950993,
        0.99980456,
        0.99990857,
        0.9998807,
        0.9999105,
        0.9999471,
        0.99986196,
        0.9998946,
        0.9998203,
        0.9998598,
        0.9996904,
        0.9990146,
        0.9973947,
        0.9050573,
        0.4101963,
        0.12207572,
        0.68449837,
        0.9734631,
        0.9999044,
        0.9999918,
        0.9999932,
        0.9999857,
        0.99998415,
        0.9998011,
        0.9997812,
        0.99964845,
        0.999997,
        0.9999982,
        0.99994326,
        0.99991846,
        0.99998474,
        0.9999969,
        0.99999034,
        0.9999832,
        0.99993634,
        0.9989524,
        0.9981692,
        0.9994461,
        0.99889356,
        0.99100083,
        0.80809134,
        0.31001306,
        0.07813915,
        0.032175343,
        0.017167768,
        0.010857119,
        0.008700228,
        0.008970643,
        0.009144668,
        0.009049293,
        0.020358618,
        0.010069266,
        0.009570069,
        0.008366508,
        0.007237987,
        0.009487299,
        0.009917064,
        0.011310629,
        0.009329891,
        0.00979836,
        0.009688303,
        0.021355843,
        0.011917363,
        0.008354234,
        0.007124569,
        0.009126776,
        0.009022113,
        0.009793601,
        0.42455977,
        0.9993143,
        0.99861026,
        0.9996804,
        0.9999511,
        0.9999391,
        0.9999751,
        0.99994016,
        0.99993527,
        0.9999386,
        0.9999535,
        0.9999496,
        0.99990964,
        0.9999155,
        0.9997744,
        0.9999161,
        0.9999298,
        0.99986315,
        0.9997621,
        0.9995703,
        0.99896705,
        0.99819297,
        0.92216605,
        0.6736925,
        0.25940087,
        0.5616268,
        0.6943272,
        0.99988973,
        0.99989784,
        0.9999565,
        0.9999894,
        0.9999777,
        0.99980575,
        0.999936,
        0.99889576,
        0.9985071,
        0.9997863,
        0.9872069,
        0.9940111,
        0.9870055,
        0.98515713,
        0.9921653,
        0.99590755,
        0.98509675,
        0.84745604,
        0.3660699,
        0.098158985,
        0.6594947,
        0.9979436,
        0.9999505,
        0.9999728,
        0.9999479,
        0.99981874,
        0.9994497,
        0.9975667,
        0.9984565,
        0.9831865,
        0.9945635,
        0.99984884,
        0.99994755,
        0.9998909,
        0.99974555,
        0.99995804,
        0.9999579,
        0.9999354,
        0.99991524,
        0.9999393,
        0.999795,
        0.99956506,
        0.9984181,
        0.9976046,
        0.99750155,
        0.99078864,
        0.6405812,
        0.17304267,
        0.039175116,
        0.012960409,
        0.0090165,
        0.0070693395,
        0.007911437,
        0.0067427726,
        0.0074840584,
        0.007057808,
        0.006615305,
        0.0075314427,
        0.0057101157,
        0.0061597987,
        0.005438509,
        0.0077358573,
        0.005126318,
        0.006356426,
    ];

    #[test]
    fn test_real_data() {
        type B = PerformanceBackend;
        type F = <B as BackendTypes>::FloatElem;

        let path = silero_burnpack_path();
        let device = Default::default();

        let s_mod: SileroVad16x8<B> =
            SileroVad16x8::load_from_burnpack(&path, &device).ok_or_panic();

        let r_mod: ReferenceVAD<B> = ReferenceVAD::from_file(path.to_str().unwrap(), &device);

        let batch = 1;
        let state = Tensor::zeros([2, batch, 128], &device);

        let sample_rate = 16000;

        let input: Vec<f32> = fs::read("input.bin")
            .unwrap()
            .chunks_exact(4)
            .map(|bytes| f32::from_le_bytes(bytes.try_into().unwrap()))
            .collect();

        let mut s_out;
        let mut r_out;
        let mut s_state = state.clone();
        let mut r_state = state.clone();

        for (i, chunk) in input.chunks_exact(512).enumerate() {
            println!("Iter {i}");
            let input = Tensor::<B, 1>::from_floats(chunk, &device).unsqueeze_dim(0);
            (s_out, s_state) = s_mod.forward(input.clone(), sample_rate, s_state);
            (r_out, r_state) = r_mod.forward(input.clone(), sample_rate, r_state);

            let correct_output = REF_DATA[i];
            let correct_output_tensor = TensorData::from([[correct_output]]);
            let s_out_data = s_out.to_data();
            let r_out_data = r_out.to_data();

            s_out_data.assert_approx_eq::<F>(&r_out_data, Tolerance::default());
            s_out_data.assert_approx_eq::<F>(&correct_output_tensor, Tolerance::default());
            r_out_data.assert_approx_eq::<F>(&correct_output_tensor, Tolerance::default());

            s_state
                .to_data()
                .assert_approx_eq::<F>(&r_state.to_data(), Tolerance::default());
        }
    }
}
