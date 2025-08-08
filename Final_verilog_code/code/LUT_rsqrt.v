module LUT_rsqrt (
    input  wire        clk,
    input  wire        rst_n,
    input  wire        in_valid,
    input  wire [15:0] x_in,
    output reg         out_valid,
    output reg  [15:0] slope_out,
    output reg  [15:0] intercept_out
);

    localparam integer N_SEGMENTS = 8;

    reg [15:0] breakpoints [0:N_SEGMENTS];
    reg [15:0] slopes      [0:N_SEGMENTS-1];
    reg [15:0] intercepts  [0:N_SEGMENTS-1];

    reg [2:0] region;  // 최대 8개 → 3비트 필요
    reg       region_valid;
    reg [2:0] next_region;

    initial begin
        breakpoints[0] = 16'h001A; // 0.1016
        breakpoints[1] = 16'h0024; // 0.1406
        breakpoints[2] = 16'h0038; // 0.2188
        breakpoints[3] = 16'h005A; // 0.3516
        breakpoints[4] = 16'h0099; // 0.5977
        breakpoints[5] = 16'h0114; // 1.0781
        breakpoints[6] = 16'h0216; // 2.0898
        breakpoints[7] = 16'h0482; // 4.5078
        breakpoints[8] = 16'h0B45; // 11.270

        slopes[0] = 16'hB541; // -75.75
        slopes[1] = 16'hDFA3; // -33.23
        slopes[2] = 16'hF339; // -12.91
        slopes[3] = 16'hFB5F; //  -4.63
        slopes[4] = 16'hFE80; //  -1.50
        slopes[5] = 16'hFF93; //  -0.43
        slopes[6] = 16'hFFE6; //  -0.10
        slopes[7] = 16'hFFFB; //  -0.02

        intercepts[0] = 16'h1164; // 17.39
        intercepts[1] = 16'h0B78; // 11.47
        intercepts[2] = 16'h0738; // 7.22
        intercepts[3] = 16'h045A; // 4.35
        intercepts[4] = 16'h027C; // 2.48
        intercepts[5] = 16'h0154; // 1.33
        intercepts[6] = 16'h00A6; // 0.65
        intercepts[7] = 16'h0047; // 0.28
    end

    // Region 결정 (combinational)
    always @(*) begin
        if      (x_in < breakpoints[1]) next_region = 0;
        else if (x_in < breakpoints[2]) next_region = 1;
        else if (x_in < breakpoints[3]) next_region = 2;
        else if (x_in < breakpoints[4]) next_region = 3;
        else if (x_in < breakpoints[5]) next_region = 4;
        else if (x_in < breakpoints[6]) next_region = 5;
        else if (x_in < breakpoints[7]) next_region = 6;
        else                            next_region = 7;
    end

    // Region 저장 (1 cycle)
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            region <= 0;
            region_valid <= 0;
        end else if (in_valid) begin
            region <= next_region;
            region_valid <= 1;
        end else begin
            region_valid <= 0;
        end
    end

    // 출력 (1 cycle 지연)
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            slope_out     <= 0;
            intercept_out <= 0;
            out_valid     <= 0;
        end else begin
            slope_out     <= slopes[region];
            intercept_out <= intercepts[region];
            out_valid     <= region_valid;
        end
    end

endmodule
