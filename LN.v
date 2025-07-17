`timescale 1ns / 1ps
// -----------------------------------------------------------------------------
//  Layer-Normalization datapath (64 × FP16)  ―  fully-wired back-pressure
// -----------------------------------------------------------------------------
module LN(
    input  wire        clk,
    input  wire        rst_n,

    input  wire        x_valid,
    input  wire [64*16-1:0] a,
    input  wire        downstream_ready,     // 최종 Stage ready

    output wire        adder_input_ready,
    output wire        out_valid,
    output wire        mean_out_valid,
    output wire        diff_vec_valid,
    output wire        squared_diff_valid,
    output wire        rsqrt_valid,

    output wire [15:0] mean_out,
    output wire [64*16-1:0] diff_vec,
    output wire [64*16-1:0] normalized_vec,
    output wire [15:0] sum_squared_diff,
    output wire [15:0] rsqrt_out,

    // Debug
    output wire [63:0] dbg_sub_valid,
    output wire [63:0] dbg_sq_valid,
    output wire [63:0] dbg_norm_valid,
    output wire        dbg_sum_valid,
    output wire        dbg_rsqrt_valid
);

// ───────────────────────────────────────────────
// Stage-0 : 입력 래치  (ready = stage1_ready)
// ───────────────────────────────────────────────
wire stage1_ready;
assign adder_input_ready = stage1_ready;

reg [64*16-1:0] reg_a;
reg             in_valid_latched;

always @(posedge clk or negedge rst_n) begin
    if(!rst_n) begin
        reg_a            <= {64*16{1'b0}};
        in_valid_latched <= 1'b0;
    end else if(x_valid && adder_input_ready) begin
        reg_a            <= a;
        in_valid_latched <= 1'b1;
    end else begin
        in_valid_latched <= 1'b0;
    end
end

// ───────────────────────────────────────────────
// Stage-1 : mean = Σ/64
// ───────────────────────────────────────────────
wire mean_valid;
wire [15:0] mean_fp16;

adder_div u_div (
    .clk              (clk),
    .rst_n            (rst_n),
    .in_valid         (in_valid_latched),
    .a_vec            (reg_a),
    .downstream_ready (stage1_ready),   // back-pressure from Stage-2
    .out_valid        (mean_valid),
    .mean             (mean_fp16)
);

assign mean_out_valid = mean_valid;
assign mean_out       = mean_fp16;

// ───────────────────────────────────────────────
// Stage-2 : diff = a[i] - mean
// ───────────────────────────────────────────────
wire [63:0] sub_a_ready, sub_b_ready;
wire [63:0] diff_valid_vec;
wire [63:0] diff_out_ready;

assign stage1_ready   = &sub_a_ready & &sub_b_ready; // to Stage-1
assign diff_vec_valid = &diff_valid_vec;
assign dbg_sub_valid  = diff_valid_vec;

genvar gi;
generate
    for(gi = 0; gi < 64; gi = gi + 1) begin : GEN_SUB
        floating_point_addsub u_sub (
            .aclk                 (clk),

            // ★ tvalid = mean_valid   (stage1_ready 제거)
            .s_axis_a_tvalid      (mean_valid),
            .s_axis_a_tdata       (reg_a[gi*16 +: 16]),
            .s_axis_a_tready      (sub_a_ready[gi]),

            .s_axis_b_tvalid      (mean_valid),
            .s_axis_b_tdata       (mean_fp16),
            .s_axis_b_tready      (sub_b_ready[gi]),

            .m_axis_result_tvalid (diff_valid_vec[gi]),
            .m_axis_result_tdata  (diff_vec[gi*16 +: 16]),
            .m_axis_result_tready (diff_out_ready[gi])   // from Stage-3
        );
    end
endgenerate

// ───────────────────────────────────────────────
// Stage-3 : square = diff²
// ───────────────────────────────────────────────
wire [63:0] sq_a_ready, sq_b_ready;
wire [63:0] sq_valid_vec;
wire [64*16-1:0] squared_vec;

assign diff_out_ready = sq_a_ready & sq_b_ready; // to Stage-2

// ★ tvalid = diff_vec_valid   (sq_*_ready AND 제거)
wire stage3_tvalid = diff_vec_valid;

assign squared_diff_valid = &sq_valid_vec;
assign dbg_sq_valid       = sq_valid_vec;

genvar gj;
generate
    for(gj = 0; gj < 64; gj = gj + 1) begin : GEN_SQ
        floating_point_mult u_sq (
            .aclk                 (clk),

            .s_axis_a_tvalid      (stage3_tvalid),
            .s_axis_a_tdata       (diff_vec[gj*16 +: 16]),
            .s_axis_a_tready      (sq_a_ready[gj]),

            .s_axis_b_tvalid      (stage3_tvalid),
            .s_axis_b_tdata       (diff_vec[gj*16 +: 16]),
            .s_axis_b_tready      (sq_b_ready[gj]),

            .m_axis_result_tvalid (sq_valid_vec[gj]),
            .m_axis_result_tdata  (squared_vec[gj*16 +: 16]),
            .m_axis_result_tready (stage4_ready)        // from Stage-5
        );
    end
endgenerate

// ───────────────────────────────────────────────
// Stage-4 : variance = mean(squared_vec)
// ───────────────────────────────────────────────
wire stage4_ready;      // = rsqrt s_axis_a_tready
wire var_valid;
wire [15:0] var_fp16;

adder_div u_var (
    .clk              (clk),
    .rst_n            (rst_n),
    .in_valid         (squared_diff_valid),
    .a_vec            (squared_vec),
    .downstream_ready (stage4_ready),
    .out_valid        (var_valid),
    .mean             (var_fp16)
);

assign sum_squared_diff = var_fp16;
assign dbg_sum_valid    = var_valid;

// ───────────────────────────────────────────────
// Stage-5 : rsqrt(variance)
// ───────────────────────────────────────────────
wire rsqrt_a_ready;
wire [63:0] norm_a_ready, norm_b_ready;
wire rsqrt_result_ready = &norm_a_ready & &norm_b_ready;

floating_point_rsqrt u_rsqrt (
    .aclk                 (clk),
    .s_axis_a_tvalid      (var_valid),
    .s_axis_a_tdata       (var_fp16),
    .s_axis_a_tready      (rsqrt_a_ready),
    .m_axis_result_tvalid (rsqrt_valid),
    .m_axis_result_tdata  (rsqrt_out),
    .m_axis_result_tready (rsqrt_result_ready)
);

assign dbg_rsqrt_valid = rsqrt_valid;
assign stage4_ready    = rsqrt_a_ready;

// ───────────────────────────────────────────────
// Stage-6 : normalize = diff × rsqrt
// ───────────────────────────────────────────────
wire [63:0] norm_valid_vec;

genvar gk;
generate
    for(gk = 0; gk < 64; gk = gk + 1) begin : GEN_NORM
        floating_point_mult u_norm (
            .aclk                 (clk),

            .s_axis_a_tvalid      (rsqrt_valid),
            .s_axis_a_tdata       (diff_vec[gk*16 +: 16]),
            .s_axis_a_tready      (norm_a_ready[gk]),

            .s_axis_b_tvalid      (rsqrt_valid),
            .s_axis_b_tdata       (rsqrt_out),
            .s_axis_b_tready      (norm_b_ready[gk]),

            .m_axis_result_tvalid (norm_valid_vec[gk]),
            .m_axis_result_tdata  (normalized_vec[gk*16 +: 16]),
            .m_axis_result_tready (downstream_ready)
        );
    end
endgenerate

assign dbg_norm_valid = norm_valid_vec;
assign out_valid      = &norm_valid_vec;

endmodule
