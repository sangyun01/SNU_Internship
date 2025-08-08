module LN_FSM (
    input  wire              clk,
    input  wire              rst_n,
    output reg               in_valid,
    input  wire              in_ready,
    output reg  [64*16-1:0]  a_in,
    input  wire              out_valid,
    output reg               out_ready
);

    reg [1:0] state, next_state;
    localparam IDLE        = 2'b00,
               SEND_INPUT  = 2'b01,
               WAIT_RESULT = 2'b10;

    reg [1:0] input_index;

    // 상태 전이
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= IDLE;
        end else begin
            state <= next_state;
        end
    end

    // 다음 상태 결정
    always @(*) begin
        case (state)
            IDLE:        next_state = in_ready ? SEND_INPUT : IDLE;
            SEND_INPUT:  next_state = WAIT_RESULT;
            WAIT_RESULT: next_state = out_valid ? IDLE : WAIT_RESULT;
            default:     next_state = IDLE;
        endcase
    end

    // 입력 벡터 정의 (4세트 예시)
    wire [1023:0] input_set0 = {
        16'hFCA6, 16'h0430, 16'h0262, 16'h04FD,
        16'hFD47, 16'h03B6, 16'hFD6E, 16'h0192,
        16'hFCF5, 16'h0388, 16'h032C, 16'h0131,
        16'hFFD6, 16'hFC34, 16'h023E, 16'h003C,
        16'h0273, 16'h03B2, 16'h04BE, 16'h039A,
        16'hFB3B, 16'hFF6C, 16'h0284, 16'hFD89,
        16'h00CC, 16'hFBF9, 16'hFDDB, 16'hFB5C,
        16'h030E, 16'hFE1C, 16'hFF4A, 16'h0446,
        16'h024D, 16'hFB9A, 16'hFD75, 16'h019F,
        16'h0142, 16'hFE41, 16'h0455, 16'h018F,
        16'h00EB, 16'h015D, 16'h0284, 16'hFEFB,
        16'hFFC2, 16'hFDF6, 16'hFDB4, 16'h046E,
        16'h0298, 16'h008B, 16'hFE26, 16'hFE6C,
        16'hFC08, 16'h0012, 16'hFEFB, 16'h023C,
        16'hFF94, 16'hFDA1, 16'hFB79, 16'hFC2C,
        16'h0218, 16'h003C, 16'h00ED, 16'h03F9
    };

    wire [1023:0] input_set1 = {
        16'h0205, 16'h0443, 16'hFFB9, 16'hFB49,
        16'h01FC, 16'hFB29, 16'hFD5F, 16'h0138,
        16'hFDB7, 16'h025B, 16'h04F3, 16'hFD3C,
        16'h00D6, 16'hFEF3, 16'h0433, 16'h01B2,
        16'h035C, 16'h029F, 16'hFB97, 16'hFE27,
        16'hFE5A, 16'hFCF6, 16'hFBF5, 16'h033B,
        16'h0228, 16'h0454, 16'h0226, 16'h031C,
        16'hFC3D, 16'hFB5F, 16'h0389, 16'hFC60,
        16'hFF80, 16'h00ED, 16'hFEC2, 16'h022F,
        16'hFBAC, 16'h0381, 16'h0266, 16'h0156,
        16'hFCA5, 16'h0039, 16'hFE75, 16'h026C,
        16'h0467, 16'h042D, 16'hFF6A, 16'hFE82,
        16'h040A, 16'h0436, 16'h00B2, 16'hFD1C,
        16'h03F1, 16'h00AC, 16'h0273, 16'h009E,
        16'h0309, 16'hFC57, 16'hFBC2, 16'hFE40,
        16'h00F1, 16'hFE24, 16'hFF4D, 16'h042C
    };

    wire [1023:0] input_set2 = {
        16'h01D0, 16'h021A, 16'h0264, 16'h02AE,
        16'h01FF, 16'h0231, 16'h0275, 16'h0220,
        16'h0255, 16'h0280, 16'h0215, 16'h0240,
        16'h027A, 16'h0250, 16'h0235, 16'h020A,
        16'h029B, 16'h0210, 16'h0225, 16'h01F5,
        16'h0265, 16'h0200, 16'h0245, 16'h0222,
        16'h025F, 16'h0272, 16'h020C, 16'h022E,
        16'h0241, 16'h025C, 16'h0273, 16'h0237,
        16'h026A, 16'h021F, 16'h01E8, 16'h0233,
        16'h024B, 16'h0204, 16'h0278, 16'h01DD,
        16'h023B, 16'h0283, 16'h0212, 16'h0259,
        16'h0290, 16'h026C, 16'h0217, 16'h0230,
        16'h01F9, 16'h0247, 16'h027D, 16'h0251,
        16'h01C5, 16'h0206, 16'h0261, 16'h0270,
        16'h025E, 16'h01E2, 16'h0211, 16'h0266,
        16'h0289, 16'h0234, 16'h025A, 16'h01FA
    };

    wire [1023:0] input_set3 = {
        16'h0208, 16'h020F, 16'h0215, 16'h021A,
        16'h0205, 16'h0210, 16'h020D, 16'h0212,
        16'h020C, 16'h0213, 16'h0216, 16'h020E,
        16'h0211, 16'h0206, 16'h0217, 16'h0209,
        16'h0207, 16'h0214, 16'h0210, 16'h020D,
        16'h020A, 16'h0212, 16'h020C, 16'h0215,
        16'h020B, 16'h0208, 16'h020E, 16'h0211,
        16'h020F, 16'h020C, 16'h0209, 16'h0213,
        16'h0216, 16'h0210, 16'h020E, 16'h020A,
        16'h0214, 16'h0211, 16'h0207, 16'h0213,
        16'h020C, 16'h0212, 16'h0208, 16'h0210,
        16'h020F, 16'h020D, 16'h0216, 16'h020A,
        16'h0213, 16'h0211, 16'h020E, 16'h0209,
        16'h020B, 16'h0215, 16'h020F, 16'h0212,
        16'h0210, 16'h0208, 16'h0214, 16'h020B,
        16'h020D, 16'h0216, 16'h020E, 16'h020C
    };

    // FSM 동작 제어
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            in_valid     <= 1'b0;
            out_ready    <= 1'b1;
            a_in         <= 0;
            input_index  <= 0;
        end else begin
            case (next_state)
                SEND_INPUT: begin
                    in_valid <= 1'b1;
                    case (input_index)
                        2'd0: a_in <= input_set0;
                        2'd1: a_in <= input_set1;
                        2'd2: a_in <= input_set2;
                        2'd3: a_in <= input_set3;
                        default: a_in <= input_set0;
                    endcase
                end
                WAIT_RESULT: begin
                    in_valid <= 1'b0;
                end
                IDLE: begin
                    if (state == WAIT_RESULT && out_valid) begin
                        input_index <= input_index + 1;
                    end
                end
                default: begin
                    in_valid <= 1'b0;
                end
            endcase
        end
    end

endmodule
