#include <torch/extension.h>
#include <torch/library.h>

using torch::Tensor;
using FuncType = std::tuple<Tensor, Tensor>(const Tensor &, int64_t,
                                            torch::optional<int64_t>,
                                            torch::optional<int64_t>,
                                            c10::string_view);

////////////////////
//                //
//    Autograd    //
//                //
////////////////////
using torch::autograd::AutogradContext;
using torch::autograd::Variable;
using torch::autograd::variable_list;
class FPSampleFunction : public torch::autograd::Function<FPSampleFunction> {
  public:
    static variable_list forward(AutogradContext *ctx, const Tensor &x,
                                 int64_t k, torch::optional<int64_t> h,
                                 torch::optional<int64_t> start_idx,
                                 c10::string_view backend) {
        torch::AutoDispatchBelowADInplaceOrView guard;
        static auto op = torch::Dispatcher::singleton()
                             .findSchemaOrThrow("torch_fpsample::sample", "")
                             .typed<FuncType>();
        auto results = op.call(x, k, h, start_idx, backend);
        auto ret_tensor = std::get<0>(results);
        auto ret_indices = std::get<1>(results);
        ctx->save_for_backward({ret_indices});
        ctx->saved_data["x_sizes"] = x.sizes();
        return {ret_tensor, ret_indices};
    }

    static variable_list backward(AutogradContext *ctx,
                                  variable_list grad_outputs) {
        auto saved_tensors = ctx->get_saved_variables();
        auto ret_indices = saved_tensors[0];
        auto x_sizes = ctx->saved_data["x_sizes"].toIntVector();
        auto grad_output = grad_outputs[0];

        auto tmp_repeat_sizes = x_sizes;
        std::fill(tmp_repeat_sizes.begin(), tmp_repeat_sizes.end() - 1, 1);

        auto grad_input = torch::scatter(
            torch::zeros(x_sizes, grad_output.options()), -2,
            ret_indices.unsqueeze(-1).repeat(tmp_repeat_sizes), grad_output);

        return {grad_input, Variable(), Variable(), Variable(), Variable()};
    }
};

std::tuple<Tensor, Tensor> sample_autograd(const Tensor &x, int64_t k,
                                           torch::optional<int64_t> h,
                                           torch::optional<int64_t> start_idx,
                                           c10::string_view backend) {
    auto results = FPSampleFunction::apply(x, k, h, start_idx, backend);
    return std::make_tuple(results[0], results[1]);
}

TORCH_LIBRARY_IMPL(torch_fpsample, Autograd, m) {
    m.impl("sample", &sample_autograd);
}
