#ifndef CADO_FACUL_STEPPER_HPP
#define CADO_FACUL_STEPPER_HPP

#include <cstddef>

#include <memory>
#include <vector>

#include "cxx_mpz.hpp"
#include "facul.hpp"
#include "facul_method.hpp"
#include "modset.hpp"

/* facul_all(), one method at a time, so that the caller gets to run some
 * of the methods somewhere else, in a batch with those of other numbers.
 * facul_all() is exactly this with nothing run elsewhere.
 *
 *      facul_stepper S(N, strategies);
 *      while (S.advance(want)) {
 *          // S.pending_method() on S.pending_modulus() yields f, or 1
 *          S.supply(f);        // or S.run_pending();
 *      }
 *      S.result();
 *
 * The factor supplied must be the one the method itself would have found,
 * and then the outcome is that of facul_all(). The methods of the second
 * stage, on the parts of a number that split, are not offered.
 */
class facul_stepper {
    public:
    /* whether this method on this modulus is to be left to the caller */
    using want_t = bool (*)(facul_method const &, FaculModulusBase const &);

    facul_stepper(std::vector<cxx_mpz> const & N,
            facul_strategies const & strategies);

    /* Runs the methods until the end, in which case it returns false, or
     * until one that want accepts, in which case it returns true. */
    bool advance(want_t want = nullptr);

    facul_method const & pending_method() const;
    FaculModulusBase const & pending_modulus() const;

    void supply(cxx_mpz const & f);
    void run_pending();

    std::vector<facul_result> & result() { return res; }

    private:
    void conclude_method(
            std::vector<std::unique_ptr<FaculModulusBase>> & more_composites);

    facul_strategies const & strategies;
    std::vector<facul_method_side> const * methods;
    std::vector<facul_result> res;
    std::vector<std::vector<std::unique_ptr<FaculModulusBase>>> composites;
    /* last_i[s] is the index of last method tried on side s */
    std::vector<size_t> last_i;
    size_t i = 0;
    bool pending = false;
    bool done = false;
#ifdef ENABLE_UNSAFE_FACUL_STATS
    int stats_nb_side = 0, stats_index_transition = 0;
#endif
};

#endif /* CADO_FACUL_STEPPER_HPP */
